"""Embedding backends.

The index stage only talks to an :class:`Embedder` protocol, so a hosted model
(Gemini) and a local one (bge, or a fake in tests) are interchangeable without
touching the build logic.

Gemini details verified live on 2026-11-14:

* model id ``gemini-embedding-001``;
* ``batchEmbedContents`` accepts a list of requests (100 per call is the
  documented ceiling) and returns one 768-dim vector per input when
  ``outputDimensionality`` is 768;
* document and query embeddings must use different ``taskType`` values, or
  retrieval quality silently degrades without any error.

The API key is read from an environment variable and never logged.
"""

from __future__ import annotations

import math
import os
import time
from typing import Protocol, Sequence

from pokerag.core.logging import get_logger
from pokerag.index.config import EmbedderConfig

logger = get_logger("pokerag.index.embedders")

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"


class EmbedderError(RuntimeError):
    """Raised when the embedding provider fails after retries."""


class Embedder(Protocol):
    """Minimal surface the index builder needs."""

    @property
    def identity(self) -> dict:
        """Provider, model, dimension and task types (for the manifest)."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk texts for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""

    def count_tokens(self, text: str) -> int:
        """Token count used for budget checks (approximate for hosted models)."""


class GeminiEmbedder:
    """Gemini embedding client with client-side pacing and bounded retries."""

    def __init__(self, cfg: EmbedderConfig, api_key: str | None = None) -> None:
        self._cfg = cfg
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise EmbedderError("GEMINI_API_KEY tidak tersedia di environment")
        from curl_cffi import requests as curl_requests

        self._session = curl_requests.Session(impersonate="chrome")
        self._last_call = 0.0
        self.requests_made = 0
        self.texts_embedded = 0

    @property
    def identity(self) -> dict:
        return {
            "provider": self._cfg.provider,
            "model": self._cfg.model,
            "dimension": self._cfg.dimension,
            "task_document": self._cfg.task_document,
            "task_query": self._cfg.task_query,
        }

    def count_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def _pace(self, items: int) -> None:
        """Respect the free-tier quota, which counts *items* per minute.

        Gemini reports ``limit: 100`` for ``embed_content_free_tier_requests``
        and counts each text, not each HTTP call; a batch of 100 therefore
        exhausts a whole minute. Pacing is expressed in items so the batch size
        can change without silently breaking the limit.
        """
        if self._cfg.items_per_minute <= 0:
            return
        interval = items * 60.0 / self._cfg.items_per_minute
        elapsed = time.monotonic() - self._last_call
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_call = time.monotonic()

    def _retry_hint(self, body: str) -> float | None:
        """Parse ``retry in 8.2s`` from a 429 payload when present."""
        import re

        match = re.search(r"retry in (\d+(?:\.\d+)?)s", body)
        return float(match.group(1)) if match else None

    def _post(self, url: str, payload: dict, items: int) -> dict:
        last_error = ""
        for attempt in range(1, self._cfg.max_retries + 1):
            self._pace(items)
            response = self._session.post(
                url,
                json=payload,
                headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json"},
                timeout=self._cfg.timeout_seconds,
            )
            self.requests_made += 1
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code in (429, 500, 502, 503, 504):
                wait = self._retry_hint(response.text) or min(120.0, 2.0**attempt * 5)
                logger.warning("embedding retry in %.0fs (%s)", wait, last_error[:120])
                time.sleep(wait + 1.0)
                continue
            raise EmbedderError(last_error)
        raise EmbedderError(f"gagal setelah {self._cfg.max_retries} percobaan: {last_error}")

    def _batch(self, texts: Sequence[str], task_type: str) -> list[list[float]]:
        url = GEMINI_ENDPOINT.format(model=self._cfg.model)
        payload = {
            "requests": [
                {
                    "model": f"models/{self._cfg.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": self._cfg.dimension,
                }
                for text in texts
            ]
        }
        data = self._post(url, payload, len(texts))
        vectors = [item["values"] for item in data.get("embeddings", [])]
        if len(vectors) != len(texts):
            raise EmbedderError(f"jumlah vektor {len(vectors)} != jumlah teks {len(texts)}")
        for vector in vectors:
            if len(vector) != self._cfg.dimension:
                raise EmbedderError(f"dimensi {len(vector)} != {self._cfg.dimension}")
        self.texts_embedded += len(texts)
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        size = max(1, self._cfg.batch_size)
        for start in range(0, len(texts), size):
            out.extend(self._batch(texts[start : start + size], self._cfg.task_document))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._batch([text], self._cfg.task_query)[0]


def make_embedder(cfg: EmbedderConfig) -> Embedder:
    """Factory used by the builder and the CLI."""
    if cfg.provider == "gemini":
        return GeminiEmbedder(cfg)
    if cfg.provider == "local":
        return LocalEmbedder(cfg)
    if cfg.provider == "st":
        return TorchEmbedder(cfg)
    raise EmbedderError(f"provider belum didukung: {cfg.provider}")


class TorchEmbedder:
    """sentence-transformers on the local GPU (CUDA), used for bge-m3.

    The model runs once, offline, and its vectors are cached; the GPU is then
    free for the generator. That is why 4 GB is workable: embedder and generator
    never have to be resident at the same time.
    """

    def __init__(self, cfg: EmbedderConfig) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self._cfg = cfg
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise EmbedderError(
                "device=cuda tetapi torch tidak melihat GPU. Pasang wheel CUDA: "
                "pip install --index-url https://download.pytorch.org/whl/cu124 torch"
            )
        self._torch = torch
        self._model = SentenceTransformer(cfg.model, device=cfg.device)
        self._model.max_seq_length = cfg.max_seq_length
        if cfg.dtype == "float16":
            self._model.half()
        self.texts_embedded = 0

    @property
    def identity(self) -> dict:
        return {
            "provider": self._cfg.provider,
            "model": self._cfg.model,
            "dimension": self._cfg.dimension,
            "task_document": "none",
            "task_query": "none",
            "device": self._cfg.device,
            "dtype": self._cfg.dtype,
            "max_seq_length": self._cfg.max_seq_length,
        }

    def count_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if vectors.shape[1] != self._cfg.dimension:
            raise EmbedderError(f"dimensi {vectors.shape[1]} != {self._cfg.dimension}")
        self.texts_embedded += len(vectors)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self._cfg.query_prefix + text])[0]


class LocalEmbedder:
    """Local ONNX embedder (fastembed), used when the hosted quota is unavailable.

    No torch dependency, so it runs on the same laptop while the 4 GB GPU is
    reserved for the generator. Its vectors live in a different cache file and
    produce a different ``index_version``, so a local index can never be
    mistaken for a Gemini one.
    """

    def __init__(self, cfg: EmbedderConfig) -> None:
        from fastembed import TextEmbedding

        self._cfg = cfg
        self._model = TextEmbedding(model_name=cfg.model)
        self.texts_embedded = 0

    @property
    def identity(self) -> dict:
        return {
            "provider": self._cfg.provider,
            "model": self._cfg.model,
            "dimension": self._cfg.dimension,
            "task_document": "none",
            "task_query": "none",
        }

    def count_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [list(map(float, vector)) for vector in self._model.embed(list(texts), batch_size=32)]
        for vector in vectors:
            if len(vector) != self._cfg.dimension:
                raise EmbedderError(f"dimensi {len(vector)} != {self._cfg.dimension}")
        self.texts_embedded += len(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self._cfg.query_prefix + text])[0]

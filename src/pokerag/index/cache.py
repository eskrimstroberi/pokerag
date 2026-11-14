"""Persistent embedding cache.

Embeddings are expensive and (for hosted models) billable, so they are stored
keyed by ``chunk_id`` plus a hash of the exact text. Re-running the index build
with unchanged chunks therefore makes **zero** API calls, and a chunk whose text
changed is re-embedded instead of silently keeping a stale vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pokerag.core.hashing import sha256_text
from pokerag.core.logging import get_logger

logger = get_logger("pokerag.index.cache")


class EmbeddingCache:
    """Parquet-backed vector store keyed by chunk id and text hash."""

    def __init__(self, path: Path, dimension: int) -> None:
        self.path = path
        self.dimension = dimension
        self._vectors: dict[str, list[float]] = {}
        self._hashes: dict[str, str] = {}
        if path.exists():
            frame = pd.read_parquet(path, columns=["chunk_id", "text_sha256", "vector"])
            self._vectors = dict(zip(frame["chunk_id"], frame["vector"]))
            self._hashes = dict(zip(frame["chunk_id"], frame["text_sha256"]))
            logger.info("cache: %s vektor dimuat dari %s", len(self._vectors), path.name)

    def get(self, chunk_id: str, text: str) -> list[float] | None:
        """Return a cached vector only when the text is unchanged."""
        if self._hashes.get(chunk_id) != sha256_text(text):
            return None
        return self._vectors.get(chunk_id)

    def put(self, chunk_id: str, text: str, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"dimensi vektor {len(vector)} != {self.dimension}")
        self._vectors[chunk_id] = vector
        self._hashes[chunk_id] = sha256_text(text)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "chunk_id": chunk_id,
                "text_sha256": self._hashes[chunk_id],
                "vector": np.asarray(self._vectors[chunk_id], dtype=np.float32),
            }
            for chunk_id in sorted(self._vectors)
        ]
        pd.DataFrame(rows).to_parquet(self.path, index=False)

    def __len__(self) -> int:
        return len(self._vectors)

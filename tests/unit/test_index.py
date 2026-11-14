"""Index identity and idempotency, with a fake embedder (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pokerag.core.jsonl import write_json
from pokerag.index.build import build_index, compute_index_version
from pokerag.index.config import IndexConfig


class FakeEmbedder:
    """Deterministic stand-in for Gemini: same text -> same vector, no calls."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls = 0
        self.texts = 0

    @property
    def identity(self) -> dict:
        return {"provider": "fake", "model": "fake-embed", "dimension": self.dimension}

    def _vector(self, text: str) -> list[float]:
        codes = [ord(char) for char in text[: self.dimension]]
        padded = codes + [0] * (self.dimension - len(codes))
        norm = sum(value * value for value in padded) ** 0.5 or 1.0
        return [value / norm for value in padded]

    def embed_documents(self, texts):
        self.calls += 1
        self.texts += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.calls += 1
        return self._vector(text)

    def count_tokens(self, text):
        return max(1, len(text) // 4)


def make_snapshot(root: Path, n: int = 3) -> None:
    snap = root / "corpus" / "tiny"
    (snap / "chunks").mkdir(parents=True)
    write_json(snap / "snapshot.json", {"snapshot_id": "snap-1", "name": "tiny"})
    docs = pd.DataFrame(
        [
            {
                "doc_id": f"d{i}",
                "url": f"https://example.invalid/{i}",
                "revid": 100 + i,
                "license_name": "CC-BY-NC-SA-2.5",
                "attribution": f"Bulbapedia, rev {100 + i}",
                "text": "x" * 100,
            }
            for i in range(n)
        ]
    )
    docs.to_parquet(snap / "documents.parquet", index=False)
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": f"c{i}",
                "doc_id": f"d{i}",
                "revid": 100 + i,
                "strategy": "heading",
                "parent_id": None,
                "role": "unit",
                "heading_path": json.dumps(["Biology"]),
                "char_start": 0,
                "char_end": 40,
                "text": f"Bulbasaur fact number {i} about grass poison",
                "n_chars": 40,
                "n_tokens": 10,
                "order": 0,
            }
            for i in range(n)
        ]
    )
    chunks.to_parquet(snap / "chunks" / "heading.parquet", index=False)


def config(root: Path) -> IndexConfig:
    return IndexConfig(
        snapshot="tiny",
        strategy="heading",
        artifacts_root=str(root),
        embedder={
            "provider": "gemini",
            "model": "fake-embed",
            "dimension": 8,
        },
    )


def test_build_stores_points_and_reuses_cache(tmp_path: Path):
    make_snapshot(tmp_path)
    cfg = config(tmp_path)
    embedder = FakeEmbedder()

    first = build_index(cfg, embedder=embedder)
    assert first["counts"]["points"] == 3
    assert first["counts"]["embedded_now"] == 3
    assert embedder.texts == 3

    second = build_index(cfg, embedder=embedder)
    assert second["cached"] is True
    assert second["counts"]["embedded_now"] == 0
    assert embedder.texts == 3, "second build must not call the embedder"


def test_index_version_tracks_inputs(tmp_path: Path):
    cfg = config(tmp_path)
    base = compute_index_version(cfg, "chunks-sha", "snap-1", None)
    assert base == compute_index_version(cfg, "chunks-sha", "snap-1", None)

    other_strategy = cfg.model_copy(update={"strategy": "fixed"})
    other_embedder = cfg.model_copy(
        update={"embedder": cfg.embedder.model_copy(update={"dimension": 16})}
    )
    assert compute_index_version(other_strategy, "chunks-sha", "snap-1", None) != base
    assert compute_index_version(other_embedder, "chunks-sha", "snap-1", None) != base
    assert compute_index_version(cfg, "chunks-sha", "snap-1", 20) != base


def test_cache_reembeds_changed_text(tmp_path: Path):
    make_snapshot(tmp_path)
    cfg = config(tmp_path)
    embedder = FakeEmbedder()
    build_index(cfg, embedder=embedder)

    chunks_path = tmp_path / "corpus" / "tiny" / "chunks" / "heading.parquet"
    chunks = pd.read_parquet(chunks_path)
    chunks.loc[0, "text"] = "Bulbasaur fact number 0 changed"
    chunks.to_parquet(chunks_path, index=False)

    rebuilt = build_index(cfg, embedder=embedder, force=True)
    assert rebuilt["counts"]["embedded_now"] == 1
    assert embedder.texts == 4

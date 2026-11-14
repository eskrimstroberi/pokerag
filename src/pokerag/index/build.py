"""Index builder: chunks -> dense vectors (Gemini) + BM25 -> Qdrant.

Identity of an index is derived from everything that can change retrieval
behaviour: the corpus snapshot, the chunk file, the embedder identity, the
distance metric and the BM25 parameters. Two different sets of parameters can
never share an ``index_version``, so a result can always be traced to an index.

Re-running the build with unchanged inputs makes zero embedding calls (cache)
and produces the same ``index_version``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pokerag.core.hashing import sha256_bytes, sha256_json
from pokerag.core.jsonl import read_json, write_json
from pokerag.core.logging import get_logger
from pokerag.index.bm25_index import build_bm25
from pokerag.index.cache import EmbeddingCache
from pokerag.index.config import IndexConfig
from pokerag.index.embedders import Embedder, make_embedder
from pokerag.index.qdrant_store import collection_count, create_collection, make_client, upsert_points

logger = get_logger("pokerag.index.build")

COLLECTION = "chunks"


def embedder_slug(cfg: IndexConfig) -> str:
    """Filesystem-safe identifier for the embedding backend."""
    name = cfg.embedder.model.replace("/", "-")
    return f"{cfg.embedder.provider}-{name}-{cfg.embedder.dimension}"


def compute_index_version(cfg: IndexConfig, chunks_sha: str, snapshot_id: str, limit: int | None) -> str:
    """Hash of every input that can change retrieval results."""
    return sha256_json(
        {
            "snapshot_id": snapshot_id,
            "chunks_sha256": chunks_sha,
            "strategy": cfg.strategy,
            "embedder": cfg.embedder.model_dump(),
            "qdrant": cfg.qdrant.model_dump(),
            "bm25": cfg.bm25.model_dump(),
            "limit": limit,
            "format": 1,
        }
    )


def _load_chunks(cfg: IndexConfig) -> pd.DataFrame:
    path = cfg.snapshot_dir / "chunks" / f"{cfg.strategy}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"chunk file tidak ada: {path}. Jalankan pokerag-chunk dulu.")
    frame = pd.read_parquet(path)
    frame = frame[frame["role"] != "parent"].copy()
    return frame.sort_values(["doc_id", "order"], kind="stable").reset_index(drop=True)


def _payloads(frame: pd.DataFrame, documents: pd.DataFrame) -> list[dict]:
    meta = {
        row.doc_id: row
        for row in documents[
            ["doc_id", "url", "revid", "license_name", "attribution"]
        ].itertuples()
    }
    payloads = []
    for row in frame.itertuples():
        document = meta.get(row.doc_id)
        payloads.append(
            {
                "chunk_id": row.chunk_id,
                "doc_id": row.doc_id,
                "revid": int(document.revid) if document is not None else int(row.revid),
                "url": document.url if document is not None else "",
                "heading_path": list(json.loads(row.heading_path)),
                "char_start": int(row.char_start),
                "char_end": int(row.char_end),
                "n_chars": int(row.n_chars),
                "license_name": document.license_name if document is not None else "",
                "attribution": document.attribution if document is not None else "",
            }
        )
    return payloads


def _embed_missing(
    frame: pd.DataFrame, cache: EmbeddingCache, embedder: Embedder, batch_size: int = 100
) -> int:
    """Embed only chunks whose text is not cached; returns the number embedded.

    The cache is flushed after every API batch, so an interrupted run loses at
    most one batch of work (important on a 4-hour free-tier budget).
    """
    pending = [
        index
        for index, row in enumerate(frame.itertuples())
        if cache.get(row.chunk_id, row.text) is None
    ]
    if not pending:
        logger.info("cache lengkap: 0 chunk perlu di-embed")
        return 0
    logger.info("%s chunk perlu di-embed (%s dari cache)", len(pending), len(frame) - len(pending))
    size = max(1, batch_size)
    for start in range(0, len(pending), size):
        indexes = pending[start : start + size]
        texts = [frame.iloc[index].text for index in indexes]
        vectors = embedder.embed_documents(texts)
        for index, vector in zip(indexes, vectors):
            row = frame.iloc[index]
            cache.put(row.chunk_id, row.text, vector)
        cache.save()
        done = min(start + size, len(pending))
        logger.info("embedded %s/%s (cache %s)", done, len(pending), len(cache))
    return len(pending)


def build_index(
    cfg: IndexConfig,
    embedder: Embedder | None = None,
    limit: int | None = None,
    force: bool = False,
) -> dict:
    """Build the dense + BM25 index and return its manifest."""
    snap = cfg.snapshot_dir
    snapshot = read_json(snap / "snapshot.json")
    chunks_path = snap / "chunks" / f"{cfg.strategy}.parquet"
    chunks_sha = sha256_bytes(chunks_path.read_bytes())
    version = compute_index_version(cfg, chunks_sha, snapshot["snapshot_id"], limit)
    output = cfg.indexes_root / version

    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        existing = read_json(manifest_path)
        count = collection_count(make_client(cfg.qdrant, output / "qdrant"), COLLECTION)
        if count == existing.get("counts", {}).get("points"):
            logger.info("index %s sudah lengkap (%s titik), 0 embed baru", version[:12], count)
            cached = dict(existing)
            cached["counts"] = dict(existing["counts"], embedded_now=0)
            cached["cached"] = True
            return cached

    frame = _load_chunks(cfg)
    if limit is not None:
        frame = frame.head(limit).copy()
    documents = pd.read_parquet(
        snap / "documents.parquet", columns=["doc_id", "url", "revid", "license_name", "attribution"]
    )

    embedder = embedder or make_embedder(cfg.embedder)
    cache = EmbeddingCache(
        Path(cfg.artifacts_root) / "embeddings" / f"{embedder_slug(cfg)}.parquet",
        cfg.embedder.dimension,
    )
    embedded_now = _embed_missing(frame, cache, embedder, batch_size=cfg.embedder.batch_size)

    vectors = np.asarray([cache.get(row.chunk_id, row.text) for row in frame.itertuples()], dtype=np.float32)
    if vectors.shape != (len(frame), cfg.embedder.dimension):
        raise RuntimeError(f"bentuk vektor {vectors.shape} tidak sesuai ({len(frame)}, {cfg.embedder.dimension})")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_unit = (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)

    output.mkdir(parents=True, exist_ok=True)
    client = make_client(cfg.qdrant, output / "qdrant")
    create_collection(client, COLLECTION, cfg.embedder.dimension, cfg.qdrant.distance)
    upsert_points(client, COLLECTION, _payloads(frame, documents), vectors_unit)
    stored = collection_count(client, COLLECTION)
    client.close()

    build_bm25(frame["text"].tolist(), cfg.bm25, output / "bm25", chunk_ids=frame["chunk_id"].tolist())

    manifest = {
        "index_version": version,
        "snapshot": cfg.snapshot,
        "snapshot_id": snapshot["snapshot_id"],
        "strategy": cfg.strategy,
        "chunks_sha256": chunks_sha,
        "embedder": cfg.embedder.model_dump(),
        "qdrant": cfg.qdrant.model_dump(),
        "bm25": cfg.bm25.model_dump(),
        "limit": limit,
        "counts": {
            "points": stored,
            "chunks": len(frame),
            "embedded_now": embedded_now,
            "cache_size": len(cache),
        },
        "vectors_sha256": sha256_bytes(vectors_unit.tobytes()),
        "files": {
            "embeddings_cache": Path(cfg.artifacts_root).joinpath("embeddings", f"{embedder_slug(cfg)}.parquet").name,
        },
    }
    write_json(manifest_path, manifest)
    _write_report(output / "index_report.md", manifest)
    logger.info("index %s: %s titik, %s chunk di-embed sekarang", version[:12], stored, embedded_now)
    return manifest


def _write_report(path: Path, manifest: dict) -> None:
    cfg = manifest
    lines = [
        "# Index report",
        "",
        f"- index_version: `{cfg['index_version']}`",
        f"- snapshot_id: `{cfg['snapshot_id']}`",
        f"- strategi chunk: `{cfg['strategy']}`",
        f"- embedder: `{cfg['embedder']['model']}` ({cfg['embedder']['dimension']} dim, "
        f"{cfg['embedder']['task_document']} / {cfg['embedder']['task_query']})",
        f"- titik Qdrant: {cfg['counts']['points']}",
        f"- BM25: k1={cfg['bm25']['k1']}, b={cfg['bm25']['b']}",
        f"- embed baru saat build ini: {cfg['counts']['embedded_now']}",
        f"- checksum vektor: `{cfg['vectors_sha256'][:16]}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

"""CLI for the index stage.

    pokerag-index build --limit 20     # smoke test
    pokerag-index build                # full build, resumable via embedding cache
    pokerag-index search "which ability does Charizard have"
    pokerag-index stats
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokerag.core.jsonl import read_json
from pokerag.index.bm25_index import load_bm25, search_bm25
from pokerag.index.build import build_index
from pokerag.index.config import IndexConfig, load_index_config
from pokerag.index.embedders import make_embedder
from pokerag.index.qdrant_store import make_client

DEFAULT_CONFIG = "configs/index.v1.yaml"


def latest_index(cfg: IndexConfig, version: str | None) -> Path:
    """Return the requested index directory, or the newest one for this snapshot."""
    root = cfg.indexes_root
    if version:
        path = root / version
        if not path.exists():
            raise SystemExit(f"index tidak ditemukan: {path}")
        return path
    candidates = []
    for manifest_path in root.glob("*/manifest.json"):
        manifest = read_json(manifest_path)
        if manifest.get("snapshot") == cfg.snapshot and manifest.get("strategy") == cfg.strategy:
            candidates.append((manifest_path.stat().st_mtime, manifest_path.parent))
    if not candidates:
        raise SystemExit("belum ada index; jalankan `pokerag-index build` dulu")
    return max(candidates)[1]


def cmd_build(args) -> int:
    cfg = load_index_config(Path(args.config))
    manifest = build_index(cfg, limit=args.limit, force=args.force)
    print(f"index_version={manifest['index_version']}")
    print(
        f"points={manifest['counts']['points']} embedded_now={manifest['counts']['embedded_now']} "
        f"strategy={manifest['strategy']} dim={manifest['embedder']['dimension']}"
    )
    return 0


def cmd_search(args) -> int:
    cfg = load_index_config(Path(args.config))
    index_dir = latest_index(cfg, args.version)
    manifest = read_json(index_dir / "manifest.json")
    embedder = make_embedder(cfg.embedder)
    vector = embedder.embed_query(args.query)

    client = make_client(cfg.qdrant, index_dir / "qdrant")
    dense = client.query_points(collection_name="chunks", query=vector, limit=args.top).points
    client.close()

    bm25, chunk_ids = load_bm25(index_dir / "bm25")
    sparse = search_bm25(bm25, chunk_ids, args.query, k=args.top)

    print(f"index {manifest['index_version'][:12]} · dim {manifest['embedder']['dimension']}")
    print("\nDENSE:")
    for hit in dense:
        payload = hit.payload
        print(f"  {hit.score:.3f}  {payload['chunk_id']}  {payload['doc_id']}  "
              f"{' > '.join(payload['heading_path'])[:60]}")
    print("\nBM25:")
    for chunk_id, score in sparse:
        print(f"  {score:.3f}  {chunk_id}")
    return 0


def cmd_stats(args) -> int:
    cfg = load_index_config(Path(args.config))
    index_dir = latest_index(cfg, args.version)
    manifest = read_json(index_dir / "manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-index", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--limit", type=int, default=None, help="batasi jumlah chunk (smoke test)")
    build.add_argument("--force", action="store_true", help="bangun ulang walau index sudah ada")

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top", type=int, default=5)
    search.add_argument("--version", default=None)

    stats = sub.add_parser("stats")
    stats.add_argument("--version", default=None)

    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    if args.command == "search":
        return cmd_search(args)
    if args.command == "stats":
        return cmd_stats(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

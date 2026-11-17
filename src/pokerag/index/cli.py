"""CLI for the index stage.

    pokerag-index build                          # semua strategi di config
    pokerag-index build --strategy heading       # satu strategi
    pokerag-index build --limit 20               # smoke test
    pokerag-index search "which ability does Charizard have"
    pokerag-index stats
    pokerag-index identity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pokerag.core.jsonl import read_json
from pokerag.index.bm25_index import load_bm25, search_bm25
from pokerag.index.build import build_all, identity_hash, truncation_summary
from pokerag.index.config import STRATEGIES, IndexConfig, load_index_config
from pokerag.index.embedders import make_embedder
from pokerag.index.identity import build_identity, load_identity, save_identity
from pokerag.index.qdrant_store import make_client

DEFAULT_CONFIG = "configs/index.gpu.yaml"


def _manifest(cfg: IndexConfig, strategy: str) -> dict:
    path = cfg.manifest_path(strategy)
    if not path.exists():
        raise SystemExit(
            f"koleksi `{cfg.collection_name(strategy)}` belum dibangun; jalankan "
            f"`pokerag-index build --strategy {strategy}` dulu"
        )
    return read_json(path)


def cmd_build(args) -> int:
    cfg = load_index_config(Path(args.config))
    if args.allow_truncation:
        cfg = cfg.model_copy(update={"allow_truncation": True})
    strategies = None if args.strategy == "all" else [args.strategy]
    print(
        f"embedder={cfg.embedder.provider}/{cfg.embedder.model} "
        f"device={cfg.embedder.device} dim={cfg.embedder.dimension} "
        f"dtype={cfg.embedder.dtype} max_seq={cfg.embedder.max_seq_length} "
        f"revision={cfg.embedder.revision}"
    )
    result = build_all(cfg, strategies=strategies, limit=args.limit, force=args.force)
    for strategy, manifest in result["manifests"].items():
        facts = result["run"][strategy]
        print(
            f"{strategy:13} collection={manifest['collection']:18} points={manifest['counts']['points']:6d} "
            f"dim={manifest['embedder']['dimension']} embedded_now={facts['embedded_now']} "
            f"build={facts['build_seconds']}s cached={facts['cached']}"
        )
    print(f"database={cfg.qdrant_dir}")
    return 0


def cmd_search(args) -> int:
    cfg = load_index_config(Path(args.config))
    manifest = _manifest(cfg, args.strategy)
    if manifest.get("embedder_identity_sha256") != identity_hash(cfg):
        raise SystemExit(
            f"koleksi `{manifest['collection']}` dibangun dengan embedder lain "
            f"({manifest.get('embedder', {}).get('model')}, "
            f"{manifest.get('embedder', {}).get('dimension')} dim). "
            "Pakai config yang sama dengan saat build; vektor query tidak boleh beda model."
        )
    embedder = make_embedder(cfg.embedder)
    vector = embedder.embed_query(args.query)

    client = make_client(cfg.qdrant, cfg.qdrant_dir)
    try:
        dense = client.query_points(
            collection_name=manifest["collection"], query=vector, limit=args.top
        ).points
    finally:
        client.close()

    bm25, chunk_ids = load_bm25(cfg.bm25_dir(args.strategy))
    sparse = search_bm25(bm25, chunk_ids, args.query, k=args.top)

    print(
        f"koleksi {manifest['collection']} · index {manifest['index_version'][:12]} · "
        f"{manifest['embedder']['model']} {manifest['embedder']['dimension']} dim"
    )
    print("\nDENSE:")
    for hit in dense:
        payload = hit.payload
        print(
            f"  {hit.score:.3f}  {payload['chunk_id']}  {payload['doc_id']}  "
            f"{' > '.join(payload['heading_path'])[:60]}"
        )
    print("\nBM25:")
    for chunk_id, score in sparse:
        print(f"  {score:.3f}  {chunk_id}")
    return 0


def cmd_stats(args) -> int:
    cfg = load_index_config(Path(args.config))
    strategies = [name for name in cfg.strategies if cfg.manifest_path(name).exists()]
    if args.strategy:
        strategies = [args.strategy]
    if not strategies:
        raise SystemExit("belum ada index; jalankan `pokerag-index build` dulu")
    payload = {name: _manifest(cfg, name) for name in strategies}
    provenance = read_json(cfg.provenance_path) if cfg.provenance_path.exists() else {}
    print(json.dumps({"manifests": payload, "provenance": provenance}, indent=2, ensure_ascii=False))
    return 0


def cmd_identity(args) -> int:
    cfg = load_index_config(Path(args.config))
    identity = load_identity(cfg)
    if args.refresh or identity is None:
        embedder = make_embedder(cfg.embedder)
        truncation = truncation_summary(cfg)
        identity = build_identity(cfg, embedder.describe(), truncation)
        save_identity(cfg, identity, cfg.models_docs_dir)
        print(f"identitas diperbarui: {cfg.identity_path}")
    print(
        json.dumps(
            {
                "model": identity["model"],
                "digest": identity["digest"],
                "architecture": identity["architecture"],
                "parameters": identity["parameters"],
                "dimension": identity["dimension"],
                "quantization": identity["quantization"],
                "context": identity["context"],
                "license": identity["license"],
                "truncation": identity["truncation"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-index", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--strategy", default="all", choices=["all", *STRATEGIES])
    build.add_argument("--limit", type=int, default=None, help="batasi jumlah chunk (smoke test)")
    build.add_argument("--force", action="store_true", help="bangun ulang walau index sudah ada")
    build.add_argument(
        "--allow-truncation",
        action="store_true",
        help="izinkan chunk melewati max_seq_length (dipotong, harus dilaporkan)",
    )

    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--strategy", default="heading", choices=list(STRATEGIES))
    search.add_argument("--top", type=int, default=5)

    stats = sub.add_parser("stats")
    stats.add_argument("--strategy", default=None, choices=list(STRATEGIES))

    identity = sub.add_parser("identity")
    identity.add_argument("--refresh", action="store_true", help="ukur ulang dari model yang dimuat")

    args = parser.parse_args(argv)
    if args.command == "build":
        return cmd_build(args)
    if args.command == "search":
        return cmd_search(args)
    if args.command == "stats":
        return cmd_stats(args)
    if args.command == "identity":
        return cmd_identity(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

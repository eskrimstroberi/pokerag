"""CLI for the chunking stage.

    pokerag-chunk build                      # all three strategies
    pokerag-chunk build --strategy heading   # one strategy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pokerag.chunking.build import build
from pokerag.chunking.config import load_chunking_config

DEFAULT_CONFIG = "configs/chunking.v1.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pokerag-chunk", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument(
        "--strategy", default="all", choices=["all", "fixed", "heading", "hierarchical"]
    )
    args = parser.parse_args(argv)

    cfg = load_chunking_config(Path(args.config))
    if args.command == "build":
        manifest = build(cfg, only=args.strategy)
        for name, entry in manifest["strategies"].items():
            print(
                f"{name:13} chunks={entry['n_chunks']:6d} p95={entry['p95_chars']:5d} "
                f"over_budget={entry['chunks_over_budget']:4d} coverage={entry['coverage_ratio']:.1%}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

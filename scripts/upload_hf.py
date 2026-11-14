"""Backup every corpus artefact to a private Hugging Face dataset repo.

Why a staging directory: Hugging Face commits per ``upload_folder`` call, so
calling ``upload_file`` 650 times would create 650 commits and be painfully
slow. Files are hard-linked (same volume) into a mirrored layout, then uploaded
in one commit. Hard links cost no extra disk space.

Layout uploaded
---------------
``register.parquet``, ``documents.parquet``, ``cleaned.jsonl``, ``chunks/*``,
``embeddings/*``, ``raw/wikitext/*``, ``raw/html/*``, ``indexes.tar.gz``,
``manifests/*``, ``README.md`` (dataset card), ``INTEGRITY.sha256``.

Usage
-----
    python scripts/upload_hf.py --repo meivawijaya/pokerag-corpus-v1 --dry-run
    python scripts/upload_hf.py --repo meivawijaya/pokerag-corpus-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

ARTIFACTS = Path("artifacts")
SNAPSHOT = ARTIFACTS / "corpus" / "pokerag-corpus-v1"
STAGING = ARTIFACTS / "hf_staging"
CARD_NAME = "README.md"

CARD = """---
license: other
license_name: cc-by-nc-sa-2.5
license_link: https://creativecommons.org/licenses/by-nc-sa/2.5/
language:
  - en
task_categories:
  - question-answering
  - text-retrieval
pretty_name: PokeRAG Corpus v1
tags:
  - rag
  - pokemon
  - bulbapedia
  - retrieval
---

# PokeRAG Corpus v1

Evidence-grounded Pokemon knowledge corpus built from **Bulbapedia** for an
academic retrieval-augmented generation project (configurations B0-B6,
ablations A1-A7). This repository is a **backup of the full pipeline state**,
including raw pages and the built index.

## Licence and attribution

* Source: [Bulbapedia](https://bulbapedia.bulbagarden.net). Article text is
  licensed **CC BY-NC-SA 2.5** (<http://creativecommons.org/licenses/by-nc-sa/2.5/>).
* Attribution: `Bulbapedia, CC BY-NC-SA 2.5, revision <revid>`. Every document
  keeps its URL, `pageid`, `revid`, revision timestamp and SHA-256 in
  `register.parquet`.
* **Non-commercial use only.** Any derivative must remain under CC BY-NC-SA 2.5
  (ShareAlike).
* Images and other fair-use media are **not** included; only article text.
* Not affiliated with or endorsed by Bulbapedia, Nintendo, Game Freak or
  The Pokemon Company.

## Contents

| Path | Description |
|---|---|
| `register.parquet` | one row per fetched page: pageid, revid, timestamp, hashes, licence |
| `documents.parquet` | 320 cleaned documents with section spans and infobox facts |
| `cleaned.jsonl` | same documents in JSONL, including cleaner version |
| `chunks/fixed.parquet` | sliding window 2048 chars / 256 overlap (baseline B2) |
| `chunks/heading.parquet` | heading-aware, 1200 chars, 120 overlap, min 240 |
| `chunks/hierarchical.parquet` | parent sections + child windows |
| `embeddings/bge-m3-1024.parquet` | chunk_id, text hash, model, dim, float32 vector |
| `raw/wikitext/` | original wikitext per revision |
| `raw/html/` | rendered HTML pinned to each revision |
| `indexes.tar.gz` | Qdrant local collection + BM25 index (rebuildable) |
| `manifests/` | snapshot id, chunking manifest, index manifest, validation report |
| `INTEGRITY.sha256` | SHA-256 of every file in this repository |

## Integrity and reproducibility

`manifests/snapshot.json` carries the `snapshot_id`: a hash over the config and
every `(doc_id, revid, content_sha256)` triple. Changing any document changes
the id, so results can be traced to an exact corpus state.

Chunking is span-anchored: each chunk stores `char_start`/`char_end` into the
document text, and the builder rejects any chunk whose text does not match that
slice. This keeps qrels valid across chunking strategies.

## Intended use

Academic experiments on retrieval quality (Recall@k, nDCG@10, MRR@10), grounded
answer quality, abstention and hallucination measurement. Not for commercial
use, and not a substitute for official Pokemon documentation.
"""

LICENSE = """PokeRAG Corpus v1 — data licence notice
========================================

The article text in this repository is derived from Bulbapedia
(https://bulbapedia.bulbagarden.net) and is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 2.5 Generic licence.

Full licence text: https://creativecommons.org/licenses/by-nc-sa/2.5/legalcode

You are free to:
  - Share: copy and redistribute the material in any medium or format
  - Adapt: remix, transform, and build upon the material

Under the following terms:
  - Attribution: give appropriate credit to Bulbapedia, provide a link to the
    licence, and indicate if changes were made. Per-document attribution is
    stored in register.parquet as
    "Bulbapedia, CC BY-NC-SA 2.5, revision <revid>".
  - NonCommercial: you may not use the material for commercial purposes.
  - ShareAlike: if you remix, transform, or build upon the material, you must
    distribute your contributions under the same licence as the original.
  - No additional restrictions: you may not apply legal terms or technological
    measures that legally restrict others from doing anything the licence
    permits.

Images, sprites and other media from Bulbapedia are NOT included in this
repository; only article text is stored.

This dataset is not affiliated with, endorsed by, or sponsored by Bulbapedia,
Nintendo, Game Freak, or The Pokemon Company. Pokemon and Pokemon character
names are trademarks of their respective owners.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, target: Path) -> None:
    """Hard-link when possible, otherwise copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if source.resolve() == target.resolve():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def pack_indexes() -> Path:
    """Pack Qdrant + BM25 directories into one archive (outside staging).

    Reuses the archive when the index tree has not changed, so retries after a
    failed upload do not spend five minutes re-gzipping.
    """
    target = ARTIFACTS / "indexes_backup.tar.gz"
    newest = max(
        (path.stat().st_mtime for path in (ARTIFACTS / "indexes").rglob("*") if path.is_file()),
        default=0,
    )
    if target.exists() and target.stat().st_mtime >= newest:
        return target
    with tarfile.open(target, "w:gz") as archive:
        for source in sorted((ARTIFACTS / "indexes").glob("*")):
            if source.is_dir():
                archive.add(source, arcname=f"indexes/{source.name}")
    return target


def stage_log(source: Path, remote: str, files: list[Path]) -> None:
    """Copy a console log as clean UTF-8 text.

    PowerShell's ``Tee-Object`` writes UTF-16LE, which Hugging Face classifies as
    a binary file and refuses to push through plain git. Decoding it and
    stripping control characters keeps the content while making it a normal text
    file.
    """
    if not source.exists():
        print(f"  lewati (tidak ada): {remote}")
        return
    raw = source.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        text = raw.decode("utf-16-le", errors="replace")
    elif raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16-be", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    clean = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    target = STAGING / remote
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(clean, encoding="utf-8")
    files.append(target)


def build_staging() -> list[Path]:
    """Mirror the artefacts into STAGING and return the file list."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    files: list[Path] = []

    def add(source: Path, remote: str) -> None:
        if not source.exists():
            print(f"  lewati (tidak ada): {remote}")
            return
        target = STAGING / remote
        link_or_copy(source, target)
        files.append(target)

    add(SNAPSHOT / "register.parquet", "register.parquet")
    add(SNAPSHOT / "documents.parquet", "documents.parquet")
    add(SNAPSHOT / "cleaned.jsonl", "cleaned.jsonl")
    add(SNAPSHOT / "selection.json", "manifests/selection.json")
    add(SNAPSHOT / "snapshot.json", "manifests/snapshot.json")
    add(SNAPSHOT / "manifest.json", "manifests/snapshot_manifest.json")
    add(SNAPSHOT / "provenance.json", "manifests/provenance.json")
    add(SNAPSHOT / "validation.json", "manifests/validation.json")
    add(SNAPSHOT / "validation_report.md", "manifests/validation_report.md")
    add(SNAPSHOT / "dedup_report.json", "manifests/dedup_report.json")
    add(SNAPSHOT / "chunks_manifest.json", "manifests/chunks_manifest.json")
    add(SNAPSHOT / "chunks_report.md", "manifests/chunks_report.md")
    add(SNAPSHOT / "fetch_log.jsonl", "manifests/fetch_log.jsonl")
    add(SNAPSHOT / "index_report.md", "manifests/index_report.md")
    stage_log(ARTIFACTS / "index_build.log", "manifests/index_gemini_build.log", files)
    stage_log(ARTIFACTS / "index_gpu_build.log", "manifests/index_gpu_build.log", files)
    stage_log(ARTIFACTS / "index_local_build.log", "manifests/index_local_build.log", files)

    for path in sorted((SNAPSHOT / "chunks").glob("*.parquet")):
        add(path, f"chunks/{path.name}")
    for path in sorted((ARTIFACTS / "embeddings").glob("*.parquet")):
        add(path, f"embeddings/{path.name}")
    for path in sorted((SNAPSHOT / "raw" / "wikitext").glob("*.json")):
        add(path, f"raw/wikitext/{path.name}")
    for path in sorted((SNAPSHOT / "raw" / "html").glob("*.html")):
        add(path, f"raw/html/{path.name}")
    for manifest_path in sorted((ARTIFACTS / "indexes").glob("*/manifest.json")):
        add(manifest_path, f"manifests/index_{manifest_path.parent.name[:12]}.json")

    add(pack_indexes(), "indexes.tar.gz")

    (STAGING / CARD_NAME).write_text(CARD, encoding="utf-8")
    files.append(STAGING / CARD_NAME)

    (STAGING / "LICENSE").write_text(LICENSE, encoding="utf-8")
    files.append(STAGING / "LICENSE")

    integrity = []
    for path in sorted(files):
        integrity.append(f"{sha256_file(path)}  {path.relative_to(STAGING).as_posix()}")
    (STAGING / "INTEGRITY.sha256").write_text("\n".join(integrity) + "\n", encoding="utf-8")
    files.append(STAGING / "INTEGRITY.sha256")
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="meivawijaya/pokerag-corpus-v1")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    who = api.whoami()
    role = who.get("auth", {}).get("accessToken", {}).get("role")
    print(f"login sebagai {who.get('name')} | tipe {who.get('type')} | token {role}")

    print("menyiapkan staging (hard link, tanpa duplikasi disk)...")
    files = build_staging()
    total = sum(path.stat().st_size for path in files)
    print(f"staging: {len(files)} file, {total / 1024 / 1024:.1f} MB")

    if args.dry_run:
        for path in sorted(files):
            print(f"  {path.stat().st_size / 1024 / 1024:8.1f} MB  {path.relative_to(STAGING).as_posix()}")
        return 0

    private = not args.public
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"repo siap: https://huggingface.co/datasets/{args.repo} (private={private})")

    api.upload_folder(
        folder_path=str(STAGING),
        repo_id=args.repo,
        repo_type="dataset",
        commit_message="PokeRAG corpus v1 backup: documents, chunks, embeddings, raw, index",
        ignore_patterns=["*.tmp", "*.lock", ".gitkeep"],
    )

    stamp = {
        "repo": args.repo,
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": len(files),
        "bytes": total,
    }
    (ARTIFACTS / "upload_manifest.json").write_text(
        json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"selesai: https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

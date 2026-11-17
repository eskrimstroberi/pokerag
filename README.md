# PokéRAG

**Evidence-Grounded Pokémon Knowledge Assistant Using Hybrid and Adaptive Retrieval**

Status: **tahap 2 — chunking & index** (retrieval end-to-end, LLM, dan evaluasi belum ada).

## Apa yang sudah ada

Pipeline korpus yang deterministik dan bisa diaudit:

```
select -> fetch -> clean -> dedup -> validate -> freeze
```

* `select` — memilih 320 halaman spesies dari National Pokédex 1–320 (bukan alfabetis).
* `fetch` — wikitext + HTML yang dipin ke satu `revid`, disimpan mentah, bisa di-resume.
* `clean` — HTML → teks bersih dengan section dan offset karakter (dasar chunking & qrels).
* `dedup` — exact hash + MinHash near-duplicate, dengan laporan keputusan.
* `validate` — 10 gate, hasilnya ditulis sebagai tabel, bukan klaim.
* `freeze` — `documents.parquet`, `register.parquet`, `snapshot_id` berbasis hash.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"      # Windows
# source .venv/bin/activate && pip install -e ".[dev]"   # Linux/WSL

set POKERAG_CONTACT_EMAIL=kamu@kampus.ac.id   # dipakai di User-Agent

pokerag-ingest select
pokerag-ingest fetch          # resumable, ~32 menit karena crawl-delay 5 s
pokerag-ingest clean
pokerag-ingest dedup
pokerag-ingest validate
pokerag-ingest freeze
```

Tes: `pytest` (40 tes, termasuk invariant offset, uji politeness tanpa jaringan,
uji idempotensi fetch/clean/freeze, dan uji index: tiga koleksi dalam satu
database, cache terikat identitas embedder, serta gate pemotongan chunk).

## Idempotensi

Semua stage aman dijalankan berulang:

| Stage | Run kedua |
|---|---|
| `fetch` | 0 request jaringan, index ditulis ulang byte-identik |
| `clean` | 0 dokumen ditulis ulang (versi + hash konfigurasi dicocokkan) |
| `dedup` / `validate` | laporan identik, urutan `doc_id` |
| `freeze` | `snapshot_id` dan hash parquet identik |

Bukti bisa diperiksa sendiri:

```bash
pokerag-ingest freeze && pokerag-ingest verify
```

`verify` menghitung ulang sha256 setiap file konten dan `snapshot_id` dari
`documents.parquet`; ia gagal bila satu byte berubah. Jam dinding dan commit git
disimpan terpisah di `provenance.json` agar tidak menggeser hash.

## Tahap 2 — chunking dan index

```bash
pokerag-chunk build                 # fixed, heading, hierarchical
pokerag-index identity              # tabel identitas embedder (dipakai build juga)
pokerag-index build                 # 3 koleksi Qdrant dalam satu database
pokerag-index search "which ability does Charizard have" --strategy heading
```

| Artefak | Isi |
|---|---|
| `chunks/<strategi>.parquet` | unit retrieval dengan `char_start`/`char_end` |
| `chunks_report.md` | jumlah chunk, p95, over-budget, cakupan |
| `docs/experiments/A1-index.md` | tabel A1 (di-commit): ukuran index, token, context length |
| `docs/models/embedding-BAAI-bge-m3.md` | identitas embedder (di-commit) |
| `artifacts/indexes/qdrant/` | satu database: `chunks_fixed`, `chunks_heading`, `chunks_hierarchical` |
| `artifacts/indexes/bm25/<strategi>/` | BM25 per koleksi, untuk B3 |
| `artifacts/indexes/<strategi>/manifest.json` | `index_version`, jumlah titik, checksum vektor |
| `artifacts/indexes/provenance.json` | waktu build, jumlah embed baru (tidak ikut hash) |
| `artifacts/embeddings/*.parquet` | cache vektor per identitas embedder |

Embedder dipin: `BAAI/bge-m3` revisi `5617a9f6…`, 1024 dim, fp16 di CUDA,
`max_seq_length: 1024`. Tiga strategi memakai fungsi embedding yang sama, jadi
perbandingan A1 hanya mengubah unit retrieval.

Batas 1024 token bukan tebakan: dengan tokenizer bge-m3, chunk `fixed` 2048
karakter adalah 597 token (p50) dan 1.005 (max). Pada 512 token, 95% chunk fixed
dipotong sebelum di-embed. Karena itu build **gagal sebelum embedding** bila ada
chunk melewati jendela (`--allow-truncation` untuk kasus yang sengaja).

Cache diikat ke identitas embedding (model, revisi, dimensi, dtype,
`max_seq_length`), sehingga build ulang dengan input sama = 0 embedding, dan
mengubah salah satu parameter tidak pernah memakai vektor lama. Terukur di
RTX 3050 4 GB: 15–32 chunk/detik, seluruh korpus (58.098 unit) sekitar 41 menit
(667 + 902 + 894 detik), puncak VRAM 1,43 GiB. Setelah build selesai proses
keluar dan VRAM kosong.

`index_version` mencakup snapshot, chunk, strategi, identitas embedder, metrik
jarak dan parameter BM25; waktu build dan jumlah embed baru disimpan terpisah
di `provenance.json` supaya manifest tetap deterministik.

### Varian index

| Config | Embedder | Dim | Koleksi | Catatan |
|---|---|---|---|---|
| `index.gpu.yaml` | `bge-m3` (CUDA, fp16, 1024 token) | 1024 | 3 | jalur utama, default CLI |
| `index.local.yaml` | `bge-small` (ONNX, CPU) | 384 | heading | fallback tanpa GPU |
| `index.v1.yaml` | Gemini API | 768 | heading | arsip; kuota free tier habis |

Ketiganya berbagi database yang sama tetapi nama koleksinya tetap per strategi;
mengganti embedder pada koleksi yang sudah ada **ditolak** tanpa `--force`.

### Backup

`python scripts/upload_hf.py --version v2` mengunggah keadaan penuh (satu
`indexes.tar.gz` berisi tiga koleksi, manifest, identitas embedder, laporan A1)
ke dataset privat `meivawijaya/pokerag-corpus-v2`. Repo v1 tetap utuh sebagai
arsip keadaan sebelumnya.

## Etika dan lisensi data

Teks Bulbapedia berlisensi **CC BY-NC-SA 2.5**. Konsekuensinya dikodekan, bukan
hanya ditulis:

* `allow_redistribute = false` → teks korpus masuk `.gitignore`, tidak pernah di-commit.
* `allow_ft_training = false` → gate B5 nanti harus memakai dataset berlisensi lain.
* Setiap dokumen menyimpan URL, `revid`, timestamp, `sha1` dari API, `sha256` teks bersih, dan atribusi.
* Crawler mematuhi `Crawl-delay: 5` dan berhenti (bukan mengakali) kalau mendapat HTTP 403.

Rincian: `docs/legal/data-license.md`.

## Struktur

```
configs/corpus.v1.yaml     seleksi, transport, lisensi, cleaning, dedup, gate
configs/chunking.v1.yaml   tiga strategi, tokenizer model, budget token
configs/index.*.yaml       embedder, database, BM25
src/pokerag/core/          hashing, JSONL, logging
src/pokerag/ingest/        select, fetch, clean, dedup, validate, freeze, cli
src/pokerag/chunking/      fixed, heading, hierarchical, tokenizer, cli
src/pokerag/index/         build, embedding cache, Qdrant, BM25, identitas, cli
tests/unit/                offset, dedup, politeness, seleksi, lisensi, index
artifacts/corpus/<name>/   artefak (diabaikan git), hanya manifest yang di-commit
docs/adr/                  keputusan teknis
docs/experiments/          tabel A1 (di-commit, hasil build)
docs/models/               identitas embedder (di-commit, hasil build)
```

## Keputusan yang masih terbuka

* Sumber data untuk LoRA B5 (korpus ini NC, jadi tidak boleh dipakai untuk training).
* Base generator B0–B5 (3B vs 1.5B) — memengaruhi seluruh rangkaian eksperimen.

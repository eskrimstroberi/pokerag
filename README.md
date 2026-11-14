# PokéRAG

**Evidence-Grounded Pokémon Knowledge Assistant Using Hybrid and Adaptive Retrieval**

Status: **tahap 1 — ingestion & preprocessing** (belum ada retrieval, LLM, atau evaluasi).

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

Tes: `pytest` (16 tes, termasuk invariant offset, uji politeness tanpa jaringan,
dan uji idempotensi: fetch kedua = 0 request, clean kedua = 0 tulisan, freeze
dua kali = hash identik).

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
pokerag-index build                 # Gemini embeddings + Qdrant + BM25
pokerag-index search "which ability does Charizard have"
```

| Artefak | Isi |
|---|---|
| `chunks/<strategi>.parquet` | unit retrieval dengan `char_start`/`char_end` |
| `chunks_report.md` | tabel A1: jumlah chunk, p95, over-budget, cakupan |
| `artifacts/indexes/<index_version>/` | Qdrant lokal, BM25, manifest, laporan |
| `artifacts/embeddings/*.parquet` | cache vektor (kunci: chunk_id + hash teks) |

Embedder dipin: `gemini-embedding-001`, 768 dim, `RETRIEVAL_DOCUMENT` untuk
dokumen dan `RETRIEVAL_QUERY` untuk query. Build ulang tidak memanggil API
(0 request) selama teks chunk tidak berubah.

### Jalur GPU (dipakai di mesin ini)

```bash
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
.venv\Scripts\python -m pip install sentence-transformers
pokerag-index --config configs/index.gpu.yaml build
```

`configs/index.gpu.yaml` memakai `BAAI/bge-m3` fp16 di CUDA (1024 dim). Terukur
pada RTX 3050 4 GB: **1.000 chunk ≈ 65 detik** (≈15 chunk/detik), jadi seluruh
korpus ≈25 menit, sekali saja. Setelah build selesai proses keluar dan VRAM
kembali kosong, sehingga generator 1.5B dapat memakai GPU yang sama tanpa
unload manual.

Tiga varian index hidup berdampingan dan tidak bisa tertukar, karena
`index_version` mencakup identitas embedder:

| Config | Embedder | Dim | Catatan |
|---|---|---|---|
| `index.gpu.yaml` | `bge-m3` (CUDA, fp16) | 1024 | jalur utama, kualitas terbaik |
| `index.local.yaml` | `bge-small` (ONNX, CPU) | 384 | fallback tanpa GPU |
| `index.v1.yaml` | Gemini API | 768 | diblokir kuota free tier |

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
src/pokerag/core/          hashing, JSONL, logging
src/pokerag/ingest/        select, fetch, clean, dedup, validate, freeze, cli
tests/unit/                invariant offset, dedup, politeness, seleksi, lisensi
artifacts/corpus/<name>/   artefak (diabaikan git), hanya manifest yang di-commit
docs/adr/                  keputusan teknis
```

## Keputusan yang masih terbuka

* Sumber data untuk LoRA B5 (korpus ini NC, jadi tidak boleh dipakai untuk training).
* Base generator B0–B5 (3B vs 1.5B) — memengaruhi seluruh rangkaian eksperimen.

# ADR-0005 — Chunking policy: fixed untuk B2–B6, heading untuk A1

**Status:** diterima
**Tanggal:** 2026-11-14

## Konteks

Brief menetapkan B2 (Naive RAG) memakai *fixed chunking*. B3–B6 tidak
menyebut perubahan chunking. Di sisi lain, data korpus sangat tidak rata:
12.981 section, median 355 karakter, 45,4% di bawah 300 karakter, 17,1% di atas
2.048 karakter (maksimum 57.710).

## Keputusan

1. **Jalur utama B2–B6 memakai `fixed`** (2.048 karakter ≈ 512 token, overlap 256).
   Alasannya metodologis: kalau chunking berubah di tengah B-series, delta
   B3−B2 mencampur efek retrieval dengan efek chunking dan tidak bisa
   diatribusikan.
2. **A1 membandingkan tiga strategi** dengan qrels identik: `fixed`,
   `heading`, `hierarchical`.
3. **Kandidat sistem terbaik: `heading`** — batas `h2`, split rekursif 1.200
   karakter, overlap 120, padding section pendek di dalam blok yang sama.
   Chunk di bawah 80 karakter dibuang karena hanya berisi tautan
   (mis. "Main article: Mew (TCG)"), bukan bukti.
4. Kalau A1 memilih `heading`, B2–B6 **dijalankan ulang** dengan chunking itu
   dan dilaporkan sebagai sistem v2 — bukan menukar chunking di satu konfigurasi.

## Konsekuensi

| | fixed | heading | hierarchical |
|---|---|---|---|
| chunk | 9.887 | 23.767 | 35.178 (23.484 child) |
| token untuk embedding | 4,99 jt | 4,84 jt | 4,67 jt (child saja) |
| ukuran vektor 1024-dim | 38,6 MB | 92,8 MB | 91,4 MB |
| cakupan teks | 100% | 99,8% | 100% |

Jumlah chunk `heading` 2,4× lebih besar, tetapi token untuk embedding hampir
sama — biaya embedding ditentukan token, bukan jumlah chunk. Yang membesar
adalah ukuran index dan jumlah kandidat rerank.

Semua chunk menyimpan `char_start`/`char_end` sehingga bisa dipetakan kembali ke
dokumen dan dipakai untuk qrels berbasis span. Invariant `chunk.text ==
doc.text[start:end]` diuji; build gagal bila dilanggar.

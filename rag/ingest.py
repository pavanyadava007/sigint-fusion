"""Chunk + embed the corpus into pgvector (idempotent: re-running replaces each source's chunks).

Corpus: markdown/text files (chunked by words with overlap, headings kept as context) and emitter catalogue JSON
(one chunk per entry, also upserted into the `emitters` table). Embeddings: BAAI/bge-small-en-v1.5 via fastembed
(384-d, ONNX, local, no API). Retrieval is hybrid (dense cosine + Postgres full-text) in agent/tools/core.py.

python ingest.py --corpus corpus/ --pg postgresql://sigint:sigint@localhost:5432/sigint
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import re

import asyncpg

from embeddings import Embedder
from retrieval import emitter_text

HERE = os.path.dirname(os.path.abspath(__file__))


def chunk_markdown(text: str, size: int = 180, overlap: int = 40):
    """Yield word-window chunks; each chunk is prefixed with the nearest heading so it stays self-describing."""
    heading, buf = "", []
    for line in text.splitlines():
        if line.startswith("#"):
            if buf:
                yield from _windows(heading, " ".join(buf), size, overlap)
                buf = []
            heading = line.lstrip("# ").strip()
        elif line.strip():
            buf.append(line.strip())
    if buf:
        yield from _windows(heading, " ".join(buf), size, overlap)


def _windows(heading: str, text: str, size: int, overlap: int):
    words = text.split()
    i = 0
    while i < len(words):
        yield (f"{heading}: " if heading else "") + " ".join(words[i : i + size])
        if i + size >= len(words):
            break
        i += size - overlap


def load_corpus(corpus_dir: str):
    """Returns (rows [(source, chunk_no, text)], emitters [dict])."""
    rows, emitters = [], []
    for f in sorted(glob.glob(os.path.join(corpus_dir, "**", "*.*"), recursive=True)):
        src = os.path.relpath(f, corpus_dir)
        if f.endswith(".json"):
            for i, e in enumerate(json.load(open(f))):
                emitters.append(e)
                rows.append((src, i, emitter_text(e)))
        elif f.endswith((".md", ".txt")):
            for i, c in enumerate(chunk_markdown(open(f, errors="ignore").read())):
                rows.append((src, i, re.sub(r"\s+", " ", c)))
    return rows, emitters


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=os.path.join(HERE, "corpus"))
    p.add_argument("--pg", default=os.getenv("PG", "postgresql://sigint:sigint@localhost:5432/sigint"))
    a = p.parse_args()
    rows, emitters = load_corpus(a.corpus)
    emb = Embedder().embed_documents([r[2] for r in rows])
    con = await asyncpg.connect(a.pg)
    await con.execute(open(os.path.join(HERE, "schema.sql")).read())
    async with con.transaction():
        for e in emitters:
            await con.execute(
                "INSERT INTO emitters(name,type,rf_min_mhz,rf_max_mhz,pri_us,pw_us,modulation,notes) VALUES($1,$2,$3,$4,$5,$6,$7,$8) "
                "ON CONFLICT (name) DO UPDATE SET type=EXCLUDED.type, rf_min_mhz=EXCLUDED.rf_min_mhz, rf_max_mhz=EXCLUDED.rf_max_mhz, "
                "pri_us=EXCLUDED.pri_us, pw_us=EXCLUDED.pw_us, modulation=EXCLUDED.modulation, notes=EXCLUDED.notes",
                e["name"], e.get("type"), e.get("rf_min_mhz"), e.get("rf_max_mhz"), e.get("pri_us"), e.get("pw_us"), e.get("modulation"), e.get("notes"))
        for src in {r[0] for r in rows}:
            await con.execute("DELETE FROM docs WHERE source=$1", src)
        await con.executemany("INSERT INTO docs(source, chunk_no, chunk, embedding) VALUES($1,$2,$3,$4)",
                              [(s, n, c, str(list(map(float, v)))) for (s, n, c), v in zip(rows, emb)])
    n = await con.fetchval("SELECT count(*) FROM docs")
    await con.close()
    print(json.dumps(dict(sources=len({r[0] for r in rows}), chunks_ingested=len(rows), emitters=len(emitters), docs_total=n)))


if __name__ == "__main__":
    asyncio.run(main())

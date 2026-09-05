"""Retrieval eval: questions.jsonl lines {"q": ..., "expect": "substring that must appear in a top-k chunk"} -> hit@k, MRR.
Runs both dense-only and hybrid so the effect of lexical fusion is measured. Writes results/rag_eval.json.

python eval.py questions.jsonl --k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime

import asyncpg

from embeddings import MODEL, Embedder
from retrieval import DENSE_SQL, search

HERE = os.path.dirname(os.path.abspath(__file__))


def _hit(rows, expect: str):
    for i, r in enumerate(rows):
        if expect.lower() in r["chunk"].lower():
            return i
    return None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("questions", nargs="?", default=os.path.join(HERE, "questions.jsonl"))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--pg", default=os.getenv("PG", "postgresql://sigint:sigint@localhost:5432/sigint"))
    p.add_argument("--out", default=os.path.join(HERE, "results", "rag_eval.json"))
    a = p.parse_args()
    emb = Embedder()
    con = await asyncpg.connect(a.pg)
    qs = [json.loads(line) for line in open(a.questions) if line.strip()]
    stats = {m: dict(hits=0, rr=0.0) for m in ("dense", "hybrid")}
    misses = []
    for q in qs:
        v = emb.embed_query(q["q"])
        dense = await con.fetch(DENSE_SQL, str(v), a.k)
        hybrid = await search(con, q["q"], v, k=a.k)
        for m, rows in (("dense", dense), ("hybrid", hybrid)):
            r = _hit(rows, q["expect"])
            if r is not None:
                stats[m]["hits"] += 1
                stats[m]["rr"] += 1 / (r + 1)
            elif m == "hybrid":
                misses.append(dict(q=q["q"], expect=q["expect"], top=[x["chunk"][:90] for x in rows[:2]]))
    n_docs = await con.fetchval("SELECT count(*) FROM docs")
    await con.close()
    res = dict(n=len(qs), k=a.k, docs=n_docs, embed_model=MODEL,
               **{f"{m}_hit_at_{a.k}": round(s["hits"] / len(qs), 3) for m, s in stats.items()},
               **{f"{m}_mrr": round(s["rr"] / len(qs), 3) for m, s in stats.items()},
               misses=misses, timestamp=datetime.now(UTC).isoformat(timespec="seconds"))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "misses"}))
    for m in misses:
        print("MISS", m)


if __name__ == "__main__":
    asyncio.run(main())

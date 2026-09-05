"""Hybrid retrieval over the `docs` table: dense cosine (pgvector HNSW) fused with Postgres full-text via reciprocal rank fusion.

Lexical retrieval catches rare literals (channel numbers, exact frequencies) that a small embedding model blurs.
"""

from __future__ import annotations

import re

import asyncpg

DENSE_SQL = "SELECT id, source, chunk, 1 - (embedding <=> $1) AS score FROM docs ORDER BY embedding <=> $1 LIMIT $2"
LEX_SQL = ("SELECT id, source, chunk, ts_rank_cd(tsv, q) AS score FROM docs, websearch_to_tsquery('english', $1) q "
           "WHERE tsv @@ q ORDER BY score DESC LIMIT $2")
NUM_SQL = "SELECT id, source, chunk, 1.0 AS score FROM docs WHERE chunk ILIKE $1 LIMIT $2"
BAND_SQL = ("SELECT id, name, type, rf_min_mhz, rf_max_mhz, pri_us, pw_us, modulation, notes FROM emitters "
            "WHERE rf_min_mhz - $2 <= $1 AND rf_max_mhz + $2 >= $1 ORDER BY (rf_max_mhz - rf_min_mhz) LIMIT 4")
CATALOGUE_SOURCE = "emitters_open.json"


def emitter_text(e: dict) -> str:
    """Canonical one-line description of a catalogue entry (used for ingest and for band-containment hits)."""
    lo, hi = e.get("rf_min_mhz"), e.get("rf_max_mhz")
    band = f"{lo}-{hi} MHz" if lo != hi else f"{lo} MHz"
    parts = [f"{e['name']} ({e.get('type')}): {band}", f"modulation {e.get('modulation')}"]
    if e.get("pri_us"):
        parts.append(f"PRI {e['pri_us']} us")
    if e.get("pw_us"):
        parts.append(f"PW {e['pw_us']} us")
    if e.get("notes"):
        parts.append(e["notes"])
    return "; ".join(parts)


def frequencies_mhz(text: str) -> list[float]:
    """Frequencies mentioned with a unit, converted to MHz: '9.4 GHz' -> 9400, '156.8 MHz' -> 156.8, '518 kHz' -> 0.518."""
    out = []
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(GHz|MHz|kHz)", text, flags=re.I):
        out.append(float(num) * {"ghz": 1e3, "mhz": 1.0, "khz": 1e-3}[unit.lower()])
    return out


def numbers_in(text: str) -> list[str]:
    """Numeric literals worth an exact match, e.g. '156.8', '1090', '9410'."""
    return [n for n in re.findall(r"\d+(?:\.\d+)?", text) if len(n.replace(".", "")) >= 3]


async def search(con: asyncpg.Connection, query: str, qvec: list[float], k: int = 5, dense_k: int = 10, lex_k: int = 5):
    dense = await con.fetch(DENSE_SQL, str(qvec), dense_k)
    lex = await con.fetch(LEX_SQL, query, lex_k)
    exact = []
    for n in numbers_in(query)[:3]:
        exact += await con.fetch(NUM_SQL, f"%{n}%", 3)
    band = []
    for f in frequencies_mhz(query)[:2]:  # catalogue entries whose band contains a mentioned frequency
        tol = max(2.0, 0.02 * f)
        band += [dict(id=-r["id"], source=CATALOGUE_SOURCE, chunk=emitter_text(dict(r))) for r in await con.fetch(BAND_SQL, f, tol)]
    fused: dict[int, dict] = {}
    for rank_list, weight in ((band, 1.2), (dense, 1.0), (lex, 1.0), (exact, 0.7)):
        for rank, r in enumerate(rank_list):
            f = fused.setdefault(r["id"], dict(id=r["id"], source=r["source"], chunk=r["chunk"], rrf=0.0))
            f["rrf"] += weight / (60 + rank)
    return sorted(fused.values(), key=lambda d: -d["rrf"])[:k]

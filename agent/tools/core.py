"""Tool implementations shared by the LangGraph agent and the MCP server. All tools are async and return strings (JSON or text)."""

from __future__ import annotations

import json
import os
import sys

import asyncpg
import httpx

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _rag in (os.path.join(_AGENT_DIR, "rag"), os.path.join(os.path.dirname(_AGENT_DIR), "rag")):  # container layout, then repo layout
    if os.path.isdir(_rag):
        sys.path.insert(0, _rag)
        break
from embeddings import Embedder  # noqa: E402
from retrieval import search  # noqa: E402

PG = os.getenv("PG", "postgresql://sigint:sigint@localhost:5432/sigint")
ML = os.getenv("ML_URL", "http://localhost:8000")
_pool: asyncpg.Pool | None = None
_emb: Embedder | None = None


async def pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(PG, min_size=1, max_size=4)
    return _pool


def embedder() -> Embedder:
    global _emb
    if _emb is None:
        _emb = Embedder()
    return _emb


def _dump(rows, **extra) -> str:
    out = [dict(r) for r in rows]
    for d in out:
        for k, v in list(d.items()):
            if hasattr(v, "isoformat"):
                d[k] = v.isoformat(timespec="seconds")
    return json.dumps(dict(count=len(out), rows=out, **extra), default=str)


async def search_docs(query: str, k: int = 5) -> str:
    """Semantic + keyword search over the SIGINT knowledge base (frequency allocations, emitter catalogue, glossary,
    analyst procedures). Cite results by their [source]."""
    async with (await pool()).acquire() as con:
        rows = await search(con, query, embedder().embed_query(query), k=k)
    if not rows:
        return "no documents found"
    return "\n---\n".join(f"[{r['source']}] {r['chunk'][:700]}" for r in rows)


async def query_emitter_db(rf_mhz: float, tolerance_mhz: float = 50) -> str:
    """Look up catalogued emitters whose band contains rf_mhz (+/- tolerance_mhz). Returns name, type, band, PRI/PW, modulation, notes."""
    rows = await (await pool()).fetch(
        "SELECT name,type,rf_min_mhz,rf_max_mhz,pri_us,pw_us,modulation,notes FROM emitters WHERE rf_min_mhz-$2<=$1 AND rf_max_mhz+$2>=$1 "
        "ORDER BY (rf_max_mhz-rf_min_mhz) ASC", rf_mhz, tolerance_mhz)
    return _dump(rows, rf_mhz=rf_mhz, tolerance_mhz=tolerance_mhz) if rows else f"no catalogue match within {tolerance_mhz} MHz of {rf_mhz} MHz"


async def recent_detections(minutes: int = 10, limit: int = 50) -> str:
    """Latest classifier (comint) and deinterleaver (resm) detections from all sensors, newest first, plus per-modulation counts."""
    p = await pool()
    rows = await p.fetch(
        "SELECT ts,sensor_id,source,rf_mhz,aoa,modulation,confidence,pw_us,pri_us,pri_type FROM detections "
        "WHERE ts>now()-make_interval(mins=>$1) ORDER BY ts DESC LIMIT $2", minutes, limit)
    counts = await p.fetch("SELECT modulation, count(*) AS n FROM detections WHERE ts>now()-make_interval(mins=>$1) GROUP BY 1 ORDER BY 2 DESC", minutes)
    return _dump(rows, minutes=minutes, per_modulation={r["modulation"]: r["n"] for r in counts}) if rows else f"no detections in the last {minutes} minutes"


async def current_tracks() -> str:
    """The fused Electronic Order of Battle: one row per live emitter track (RF, bearing, sources, sensors, hits, modulation, PRI type)."""
    rows = await (await pool()).fetch(
        "SELECT track_id,rf_mhz,aoa,sources,sensors,hits,modulation,label_agreement,pw_us,pri_us,pri_type,first_seen,last_seen FROM tracks "
        "WHERE updated > now() - interval '5 minutes' ORDER BY hits DESC")
    return _dump(rows) if rows else "no live tracks"


async def classify_signal(i: list[float], q: list[float]) -> str:
    """Classify a raw I/Q sample (>=64 samples) with the deep-learning modulation model. Returns top predictions with probabilities."""
    async with httpx.AsyncClient(timeout=30, trust_env=False) as c:
        return (await c.post(f"{ML}/classify", json=dict(i=i, q=q))).text


async def deinterleave(pdw: list[list[float]]) -> str:
    """Cluster pulse descriptor words [[toa_ms, rf_mhz, pw_us, aoa_deg], ...] into radar emitters with PRI statistics."""
    async with httpx.AsyncClient(timeout=30, trust_env=False) as c:
        return (await c.post(f"{ML}/deinterleave", json=dict(pdw=pdw))).text


async def save_report(question: str, report: str) -> str:
    """Persist a finished intelligence report (markdown with sections Summary / Emitters / Assessment / Confidence / Recommended actions)."""
    rid = await (await pool()).fetchval("INSERT INTO reports(question, report) VALUES($1,$2) RETURNING id", question, report)
    return f"saved report id={rid}"


async def list_reports(limit: int = 5) -> str:
    """List previously saved intelligence reports (newest first)."""
    rows = await (await pool()).fetch("SELECT id, ts, question, left(report, 300) AS preview FROM reports ORDER BY ts DESC LIMIT $1", limit)
    return _dump(rows) if rows else "no reports saved yet"


TOOLS = [search_docs, query_emitter_db, recent_detections, current_tracks, classify_signal, deinterleave, save_report, list_reports]

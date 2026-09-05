"""FastAPI chat endpoint for the operator console. uvicorn api:app --port 8001"""

from __future__ import annotations

import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import LLM_BASE_URL, LLM_MODEL, ask, build
from tools import core

log = logging.getLogger("agent")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
import asyncio  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402


async def _warm_up():
    """Load the LLM (Ollama loads weights on first request) and the embedding model so the first operator question is fast."""
    try:
        await asyncio.get_running_loop().run_in_executor(None, core.embedder)
        async with httpx.AsyncClient(timeout=300, trust_env=False) as c:
            await c.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'ollama')}"},
                         json=dict(model=LLM_MODEL, messages=[dict(role="user", content="ping")], max_tokens=1))
        log.info("warm-up done")
    except Exception as e:  # noqa: BLE001
        log.warning("warm-up failed: %s", e)


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_warm_up())
    yield
    task.cancel()


app = FastAPI(title="SIGINT-Fusion Agent", version="1.0.0", lifespan=lifespan)
origins = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
graph = build()


class Q(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    thread: str = Field("ui", max_length=64)


@app.post("/ask")
async def _ask(q: Q):
    try:
        res = await ask(graph, q.question, q.thread)
    except Exception as e:  # noqa: BLE001
        log.exception("agent failure")
        raise HTTPException(502, f"agent failed: {type(e).__name__}: {e}") from e
    log.info("thread=%s tools=%s seconds=%s", q.thread, [t["tool"] for t in res["trace"]], res["seconds"])
    return res


@app.get("/reports")
async def reports(limit: int = Query(20, ge=1, le=200)):
    rows = await (await core.pool()).fetch("SELECT id, ts, question, report, thread FROM reports ORDER BY ts DESC LIMIT $1", limit)
    return [dict(r, ts=r["ts"].isoformat()) for r in rows]


@app.get("/health")
async def health():
    reachable, detail = False, None
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=3, trust_env=False) as c:
            r = await c.get(f"{LLM_BASE_URL.rstrip('/')}/models", headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'ollama')}"})
            reachable = r.status_code == 200
            if reachable:
                ids = [m.get("id") for m in r.json().get("data", [])]
                detail = f"{len(ids)} models" + ("" if LLM_MODEL in ids or not ids else f"; {LLM_MODEL} NOT among them")
            else:
                detail = f"HTTP {r.status_code}"
    except Exception as e:  # noqa: BLE001
        detail = str(e)
    return dict(status="ok" if reachable else "degraded", llm=dict(base_url=LLM_BASE_URL, model=LLM_MODEL, reachable=reachable, detail=detail,
                                                                    probe_ms=round((time.time() - t0) * 1e3)))

"""SIGINT-Fusion ML service: ONNX modulation classifier, classical DSP endpoints and dashboard feeds.

Run: uvicorn app:app --port 8000
Env: MODEL=ckpt/ft.onnx  PG=postgresql://...  CORS_ORIGINS=http://localhost:5173  ORT_PROVIDERS=CPUExecutionProvider
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager

import asyncpg
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, model_validator

from sigproc.dsp import ca_cfar, deinterleave_pdw, psd, spectrogram

VERSION = "1.0.0"
MODEL = os.getenv("MODEL", "ckpt/ft.onnx")
PG = os.getenv("PG", "postgresql://sigint:sigint@localhost:5432/sigint")
PROVIDERS = os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(",")
log = logging.getLogger("ml-service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")

REQ = Counter("ml_requests_total", "requests", ["endpoint", "status"])
LAT = Histogram("ml_inference_seconds", "model inference time", buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5))


class Model:
    """Lazy-loaded ONNX session. The service starts without a model (health reports degraded) so the stack can boot before training."""

    def __init__(self, path: str):
        self.path, self.sess, self.classes, self.error = path, None, [], None

    def load(self) -> bool:
        try:
            import onnxruntime as ort

            self.sess = ort.InferenceSession(self.path, providers=PROVIDERS)
            self.classes = json.load(open(self.path + ".classes.json"))
            self.error = None
            log.info("model loaded: %s (%d classes, %s)", self.path, len(self.classes), self.sess.get_providers())
            return True
        except Exception as e:  # noqa: BLE001
            self.sess, self.error = None, str(e)
            log.warning("model not loaded: %s", e)
            return False

    @property
    def loaded(self) -> bool:
        return self.sess is not None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: [B,2,L] float32 -> logits [B,C]."""
        if not self.loaded and not self.load():
            raise HTTPException(503, f"model unavailable: {self.error}")
        x = x / (np.sqrt((x**2).sum(axis=(1, 2), keepdims=True) / x.shape[2]) + 1e-8)
        with LAT.time():
            return self.sess.run(None, {"iq": x.astype(np.float32)})[0]


model = Model(MODEL)
state: dict = {"pool": None, "t0": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load()
    try:
        state["pool"] = await asyncpg.create_pool(PG, min_size=1, max_size=8, timeout=5)
    except Exception as e:  # noqa: BLE001
        log.warning("postgres unavailable at startup: %s", e)
    yield
    if state["pool"]:
        await state["pool"].close()


app = FastAPI(title="SIGINT-Fusion ML Service", version=VERSION, lifespan=lifespan)
origins = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


async def db():
    if state["pool"] is None:
        try:
            state["pool"] = await asyncpg.create_pool(PG, min_size=1, max_size=8, timeout=5)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(503, f"database unavailable: {e}") from e
    return state["pool"]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ------------------------------------------------------------------ schemas
class IQ(BaseModel):
    i: list[float] = Field(..., min_length=64, max_length=65536)
    q: list[float] = Field(..., min_length=64, max_length=65536)
    top_k: int = Field(5, ge=1, le=24)

    @model_validator(mode="after")
    def _same_len(self):
        if len(self.i) != len(self.q):
            raise ValueError("i/q length mismatch")
        return self

    def array(self) -> np.ndarray:
        x = np.array([self.i, self.q], np.float32)
        if not np.isfinite(x).all():
            raise HTTPException(400, "non-finite sample values")
        return x


class PDW(BaseModel):
    pdw: list[list[float]] = Field(..., min_length=1, max_length=20000)


class SynthReq(BaseModel):
    modulation: str = "QPSK"
    snr_db: float = Field(10, ge=-20, le=30)
    length: int = Field(1024, ge=64, le=8192)
    seed: int = 0


# ------------------------------------------------------------------ inference + DSP
@app.get("/health")
def health():
    return dict(status="ok" if model.loaded else "degraded", model=model.path, model_loaded=model.loaded, error=model.error,
                classes=len(model.classes), providers=model.sess.get_providers() if model.loaded else [],
                uptime_s=round(time.time() - state["t0"], 1), version=VERSION)


@app.get("/modulations")
def modulations():
    if not model.loaded:
        model.load()
    return model.classes


@app.post("/classify")
def classify(s: IQ):
    x = s.array()
    t0 = time.perf_counter()
    p = softmax(model.predict(x[None])[0])
    dt = (time.perf_counter() - t0) * 1e3
    idx = p.argsort()[::-1][: s.top_k]
    REQ.labels("classify", "200").inc()
    return dict(predictions=[dict(modulation=model.classes[i], prob=round(float(p[i]), 4)) for i in idx], latency_ms=round(dt, 2), n_samples=len(s.i))


@app.post("/detect")
def detect(s: IQ):
    f, P = psd(s.array())
    mask = ca_cfar(P)
    REQ.labels("detect", "200").inc()
    return dict(freqs=f.round(4).tolist(), levels_db=P.round(2).tolist(), mask=mask.tolist(), n_detections=int(mask.sum()))


@app.post("/spectrogram")
def spectrogram_ep(s: IQ):
    f, t, S = spectrogram(s.array())
    return dict(freqs=f.round(4).tolist(), times=t.round(2).tolist(), db=S.round(1).tolist())


@app.post("/deinterleave")
def deinterleave(s: PDW):
    arr = np.array(s.pdw, float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise HTTPException(400, "pdw rows must be [toa_ms, rf_mhz, pw_us, aoa_deg]")
    labels, emitters = deinterleave_pdw(arr)
    REQ.labels("deinterleave", "200").inc()
    return dict(labels=labels.tolist(), emitters={str(k): v for k, v in emitters.items()})


@app.post("/synth")
def synth(r: SynthReq):
    from data.radioml import MODS_2018
    from data.synth_mod import make_sample

    if r.modulation not in MODS_2018:
        raise HTTPException(400, f"unknown modulation; choose one of {MODS_2018}")
    x = make_sample(r.modulation, r.length, r.snr_db, r.seed)
    return dict(i=x[:, 0].round(5).tolist(), q=x[:, 1].round(5).tolist(), modulation=r.modulation, snr_db=r.snr_db, synthetic=True)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ------------------------------------------------------------------ dashboard feeds (Postgres)
def _row(r) -> dict:
    d = dict(r)
    for k in ("ts", "updated", "first_seen", "last_seen"):
        if d.get(k) is not None:
            d[k] = d[k].isoformat()
    return d


@app.get("/detections")
async def detections(limit: int = Query(50, ge=1, le=1000), minutes: int = Query(60, ge=1, le=100000), sensor: str | None = None):
    pool = await db()
    q = ("SELECT id,ts,sensor_id,source,rf_mhz,aoa,modulation,confidence,latency_ms,pw_us,pri_us,pri_type FROM detections "
         "WHERE ts > now() - make_interval(mins => $1)")
    args: list = [minutes]
    if sensor:
        q += " AND sensor_id = $2"
        args.append(sensor)
    q += f" ORDER BY ts DESC LIMIT ${len(args) + 1}"
    return [_row(r) for r in await pool.fetch(q, *args, limit)]


@app.get("/sensors")
async def sensors():
    pool = await db()
    rows = await pool.fetch(
        "SELECT sensor_id, max(ts) AS last_seen, count(*) FILTER (WHERE ts > now() - interval '10 minutes') AS detections_10m, "
        "mode() WITHIN GROUP (ORDER BY source) AS kind FROM detections WHERE ts > now() - interval '1 day' GROUP BY sensor_id ORDER BY sensor_id")
    return [_row(r) for r in rows]


@app.get("/spectrum")
async def spectrum(sensor: str | None = None):
    pool = await db()
    q = "SELECT sensor_id, ts, rf_mhz, levels FROM spectra" + (" WHERE sensor_id=$1" if sensor else "") + " ORDER BY ts DESC LIMIT 1"
    r = await pool.fetchrow(q, sensor) if sensor else await pool.fetchrow(q)
    if not r:
        raise HTTPException(404, "no spectra yet")
    return _row(r)


@app.get("/spectrum/history")
async def spectrum_history(sensor: str | None = None, n: int = Query(60, ge=1, le=500)):
    pool = await db()
    if not sensor:
        r = await pool.fetchrow("SELECT sensor_id FROM spectra ORDER BY ts DESC LIMIT 1")
        if not r:
            return dict(sensor_id=None, rows=[])
        sensor = r["sensor_id"]
    rows = await pool.fetch("SELECT ts, rf_mhz, levels FROM spectra WHERE sensor_id=$1 ORDER BY ts DESC LIMIT $2", sensor, n)
    return dict(sensor_id=sensor, rows=[_row(r) for r in reversed(rows)])


@app.get("/eob")
async def eob():
    pool = await db()
    rows = await pool.fetch("SELECT * FROM tracks WHERE updated > now() - interval '5 minutes' ORDER BY hits DESC, track_id")
    return [_row(r) for r in rows]


@app.get("/stats")
async def stats():
    pool = await db()
    tot = await pool.fetchrow(
        "SELECT count(*) AS total, count(*) FILTER (WHERE ts > now() - interval '10 minutes') AS recent, "
        "count(*) FILTER (WHERE ts > now() - interval '10 minutes' AND confidence < 0.6) AS low FROM detections")
    per_mod = await pool.fetch("SELECT modulation, count(*) AS n FROM detections WHERE ts > now() - interval '10 minutes' GROUP BY 1 ORDER BY 2 DESC")
    per_sensor = await pool.fetch("SELECT sensor_id, count(*) AS n FROM detections WHERE ts > now() - interval '10 minutes' GROUP BY 1 ORDER BY 1")
    lat = await pool.fetchrow(
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50, percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95, "
        "percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99 FROM detections WHERE ts > now() - interval '10 minutes' AND latency_ms IS NOT NULL")
    tracks = await pool.fetchval("SELECT count(*) FROM tracks WHERE updated > now() - interval '5 minutes'")
    return dict(detections_total=tot["total"], detections_10m=tot["recent"], low_confidence_10m=tot["low"],
                per_modulation={r["modulation"]: r["n"] for r in per_mod}, per_sensor={r["sensor_id"]: r["n"] for r in per_sensor},
                tracks=tracks, latency_ms={k: (round(lat[k], 3) if lat[k] is not None else None) for k in ("p50", "p95", "p99")})


@app.get("/reports")
async def reports(limit: int = Query(20, ge=1, le=200)):
    pool = await db()
    return [_row(r) for r in await pool.fetch("SELECT id, ts, question, report, thread FROM reports ORDER BY ts DESC LIMIT $1", limit)]

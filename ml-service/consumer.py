"""Kafka -> inference/deinterleaving -> fusion -> Postgres.

Topics: iq-chunks (I/Q captures -> ONNX classifier -> `detections` source=comint, plus `spectra` for the waterfall)
        pdw-batches (pulse descriptor words -> DBSCAN deinterleaver -> `detections` source=resm)
Both feed the Kalman/Hungarian fuser; the live Electronic Order of Battle is upserted into `tracks`.

Batches are sized to model latency and max_poll_interval_ms is raised so heavy inference never triggers a rebalance.
Env: KAFKA=kafka:9092  PG=postgresql://...  MODEL=ckpt/ft.onnx  GROUP=ml-consumer
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from datetime import UTC, datetime

import asyncpg
import numpy as np
from aiokafka import AIOKafkaConsumer

from fusion.associate import Fuser
from sigproc.dsp import deinterleave_pdw, psd

log = logging.getLogger("ml-consumer")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
MODEL = os.getenv("MODEL", "ckpt/ft.onnx")
KAFKA = os.getenv("KAFKA", "localhost:9092")
PG = os.getenv("PG", "postgresql://sigint:sigint@localhost:5432/sigint")
SPECTRA_KEEP = int(os.getenv("SPECTRA_KEEP", "120"))  # rows kept per sensor for the waterfall
HEARTBEAT = os.getenv("HEARTBEAT", "/tmp/consumer.heartbeat")  # touched every poll; the container health check reads its age


class Classifier:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(","))
        self.classes = json.load(open(path + ".classes.json"))

    def __call__(self, x: np.ndarray):
        """x:[B,2,L] -> (labels, confidences, per-sample latency ms)."""
        x = x / (np.sqrt((x**2).sum(axis=(1, 2), keepdims=True) / x.shape[2]) + 1e-8)
        t0 = time.perf_counter()
        logits = self.sess.run(None, {"iq": x.astype(np.float32)})[0]
        dt = (time.perf_counter() - t0) * 1e3 / len(x)
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        k = p.argmax(1)
        return [self.classes[int(i)] for i in k], p[np.arange(len(k)), k].astype(float), dt


def process_iq(msgs: list[dict], clf: Classifier):
    """Pure function: I/Q messages -> (detection rows, spectra rows, fusion detections)."""
    lengths = {len(m["i"]) for m in msgs}
    dets, spectra, fus = [], [], []
    for L in lengths:  # group by length so one ONNX call handles each batch
        group = [m for m in msgs if len(m["i"]) == L]
        x = np.array([[m["i"], m["q"]] for m in group], np.float32)
        labels, conf, dt = clf(x)
        for m, lab, c, xi in zip(group, labels, conf, x):
            sid = m.get("sensor_id", "unknown")
            dets.append((sid, "comint", m.get("rf_mhz"), m.get("aoa"), lab, float(c), float(dt)))
            spectra.append((sid, m.get("rf_mhz"), psd(xi)[1].astype(float).tolist()))
            if m.get("rf_mhz") is not None and m.get("aoa") is not None:
                fus.append(dict(rf_mhz=float(m["rf_mhz"]), aoa=float(m["aoa"]), source="comint", sensor_id=sid, label=lab))
    return dets, spectra, fus


def process_pdw(msgs: list[dict]):
    """Pure function: PDW batch messages -> (detection rows, fusion detections)."""
    dets, fus = [], []
    for m in msgs:
        pdw = np.array(m.get("pdw", []), float)
        if pdw.ndim != 2 or pdw.shape[1] != 4 or len(pdw) < 5:
            continue
        sid = m.get("sensor_id", "unknown")
        t0 = time.perf_counter()
        _, emitters = deinterleave_pdw(pdw)
        dt = (time.perf_counter() - t0) * 1e3
        for e in emitters.values():
            pri_us = e["pri_mean"] * 1e3 if e["pri_mean"] else None
            dets.append((sid, "resm", e["rf"], e["aoa"], "pulse", 1.0, dt, e["pw"], pri_us, e["pri_type"]))
            fus.append(dict(rf_mhz=e["rf"], aoa=e["aoa"], source="resm", sensor_id=sid, label="pulse", pw_us=e["pw"], pri_us=pri_us, pri_type=e["pri_type"]))
    return dets, fus


def _beat() -> None:
    try:
        with open(HEARTBEAT, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


async def connect_pg(retries: int = 30):
    for i in range(retries):
        try:
            return await asyncpg.create_pool(PG, min_size=1, max_size=4)
        except Exception as e:  # noqa: BLE001
            log.warning("postgres not ready (%s), retry %d", e, i)
            await asyncio.sleep(2)
    raise RuntimeError("postgres unreachable")


async def write(pool, dets, spectra, eob, sensors_touched):
    async with pool.acquire() as con, con.transaction():
        if dets:
            await con.executemany(
                "INSERT INTO detections(sensor_id,source,rf_mhz,aoa,modulation,confidence,latency_ms,pw_us,pri_us,pri_type) "
                "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)", [d + (None,) * (10 - len(d)) for d in dets])
        if spectra:
            await con.executemany("INSERT INTO spectra(sensor_id,rf_mhz,levels) VALUES($1,$2,$3)", spectra)
            for sid in sensors_touched:
                await con.execute("DELETE FROM spectra WHERE sensor_id=$1 AND id < "
                                  "(SELECT id FROM spectra WHERE sensor_id=$1 ORDER BY id DESC OFFSET $2 LIMIT 1)", sid, SPECTRA_KEEP)
        if eob is not None:
            ts = lambda t: datetime.fromtimestamp(t, tz=UTC)  # noqa: E731
            await con.executemany(
                "INSERT INTO tracks(track_id,updated,first_seen,last_seen,rf_mhz,aoa,rf_rate,aoa_rate,sources,sensors,hits,misses,"
                "modulation,label_agreement,pw_us,pri_us,pri_type) VALUES($1,now(),$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16) "
                "ON CONFLICT (track_id) DO UPDATE SET updated=now(), last_seen=EXCLUDED.last_seen, rf_mhz=EXCLUDED.rf_mhz, aoa=EXCLUDED.aoa, "
                "rf_rate=EXCLUDED.rf_rate, aoa_rate=EXCLUDED.aoa_rate, sources=EXCLUDED.sources, sensors=EXCLUDED.sensors, hits=EXCLUDED.hits, "
                "misses=EXCLUDED.misses, modulation=EXCLUDED.modulation, label_agreement=EXCLUDED.label_agreement, pw_us=EXCLUDED.pw_us, "
                "pri_us=EXCLUDED.pri_us, pri_type=EXCLUDED.pri_type",
                [(t["track_id"], ts(t["first_seen"]), ts(t["last_seen"]), t["rf_mhz"], t["aoa"], t["rf_rate"], t["aoa_rate"], t["sources"], t["sensors"],
                  t["hits"], t["misses"], t["modulation"], t["label_agreement"], t.get("pw_us"), t.get("pri_us"), t.get("pri_type")) for t in eob])
            live = [t["track_id"] for t in eob]
            await con.execute("DELETE FROM tracks WHERE NOT (track_id = ANY($1::int[]))", live)


async def main():
    clf = Classifier(MODEL)
    log.info("model %s loaded (%d classes)", MODEL, len(clf.classes))
    pool = await connect_pg()
    fuser = Fuser(gate=float(os.getenv("FUSION_GATE", "4.0")), max_misses=int(os.getenv("FUSION_MAX_MISSES", "20")))
    consumer = AIOKafkaConsumer("iq-chunks", "pdw-batches", bootstrap_servers=KAFKA, group_id=os.getenv("GROUP", "ml-consumer"),
                                max_poll_interval_ms=60000, max_poll_records=64, value_deserializer=json.loads,
                                auto_offset_reset="latest", enable_auto_commit=True)
    for i in range(30):
        try:
            await consumer.start()
            break
        except Exception as e:  # noqa: BLE001
            log.warning("kafka not ready (%s), retry %d", e, i)
            await asyncio.sleep(2)
    else:
        raise RuntimeError("kafka unreachable")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, stop.set)
    log.info("consuming from %s", KAFKA)
    n_total = 0
    try:
        while not stop.is_set():
            batch = await consumer.getmany(timeout_ms=200, max_records=64)
            _beat()
            if not batch:
                continue
            iq = [m.value for tp, ms in batch.items() if tp.topic == "iq-chunks" for m in ms]
            pdw = [m.value for tp, ms in batch.items() if tp.topic == "pdw-batches" for m in ms]
            dets, spectra, fus = process_iq(iq, clf) if iq else ([], [], [])
            d2, f2 = process_pdw(pdw) if pdw else ([], [])
            dets += d2
            fus += f2
            groups: dict[str, list] = {}
            for d in fus:  # one association pass per sensor so multi-sensor sightings fuse into one track
                groups.setdefault(d["sensor_id"], []).append(d)
            eob = fuser.step_multi(list(groups.values()), time.time()) if fus else None
            await write(pool, dets, spectra, eob, {s[0] for s in spectra})
            n_total += len(dets)
            if n_total % 500 < len(dets):
                log.info("detections=%d tracks=%d", n_total, len(fuser.tracks))
    finally:
        await consumer.stop()
        await pool.close()
        log.info("stopped")


if __name__ == "__main__":
    asyncio.run(main())

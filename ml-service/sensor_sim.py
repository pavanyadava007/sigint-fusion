"""Simulated edge sensors streaming into the ingest gateway.

A fixed scenario of emitters (each with RF, bearing, modulation and, for radars, PRI/PW behaviour) is observed by several
COMINT sensors (I/Q captures) and one R-ESM sensor (pulse descriptor words). Bearings drift slowly so the fusion tracker
has something to track. I/Q frames come from data/sim_frames.npz (REAL RadioML 2018.01A frames extracted by
scripts/make_sim_frames.py) when that file exists, otherwise from the synthetic modulator.

python sensor_sim.py --gateway http://localhost:8080 --rate 5 --sensors 3
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import httpx
import numpy as np

from data.synth_mod import make_sample

log = logging.getLogger("sensor-sim")


class FrameSource:
    """Yields I/Q frames [L,2] for a modulation: real RadioML frames from an npz bank if available, else synthetic."""

    def __init__(self, path: str, rng: np.random.Generator):
        self.rng, self.bank = rng, None
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=False)
            classes = [str(c) for c in z["classes"]]
            self.bank = {c: z["X"][z["y"] == i].astype(np.float32) for i, c in enumerate(classes)}
            log.info("using %d real frames from %s (%s)", len(z["y"]), path, str(z["source"]))
        else:
            log.info("no frame bank at %s, streaming synthetic frames", path)

    def __call__(self, mod: str, length: int, snr: float) -> np.ndarray:
        if self.bank and mod in self.bank:
            return self.bank[mod][self.rng.integers(len(self.bank[mod]))].T  # [2,L] -> [L,2]
        return make_sample(mod, length, snr, int(self.rng.integers(0, 2**31 - 1)))

# rf MHz, bearing deg, modulation, snr dB, radar (pw us, pri ms, pri type) or None
SCENARIO = [
    dict(name="marine VHF ch16", rf=156.8, aoa=42.0, mod="FM", snr=18, radar=None),
    dict(name="airband tower", rf=121.5, aoa=305.0, mod="AM-DSB-WC", snr=14, radar=None),
    dict(name="SSR reply", rf=1090.0, aoa=200.0, mod="OOK", snr=12, radar=(0.5, 1.0, "jitter")),  # seen by COMINT (OOK) and R-ESM (pulses) -> fused track
    dict(name="data link", rf=2400.0, aoa=95.0, mod="QPSK", snr=8, radar=None),
    dict(name="S-band surveillance radar", rf=2800.0, aoa=270.0, mod="pulse", snr=20, radar=(1.0, 1.0, "constant")),
    dict(name="X-band nav radar", rf=9410.0, aoa=118.0, mod="pulse", snr=20, radar=(0.5, 0.5, "stagger")),
    dict(name="weather radar", rf=5600.0, aoa=15.0, mod="pulse", snr=20, radar=(2.0, 1.2, "jitter")),
]


def pdw_batch(rng: np.random.Generator, t0_ms: float, duration_ms: float = 40.0) -> list[list[float]]:
    rows = []
    for e in SCENARIO:
        if not e["radar"]:
            continue
        pw, pri, kind = e["radar"]
        t, k = t0_ms + rng.uniform(0, pri), 0
        while t < t0_ms + duration_ms:
            step = {"constant": pri, "stagger": pri * (1 + 0.3 * (k % 3)), "jitter": pri * rng.uniform(0.85, 1.15)}[kind]
            rows.append([t, e["rf"] + rng.normal(0, 0.5), pw * rng.uniform(0.98, 1.02), (e["aoa"] + rng.normal(0, 1.5)) % 360])
            t += step
            k += 1
    rows.sort()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gateway", default="http://localhost:8080")
    p.add_argument("--rate", type=float, default=5, help="scenario ticks per second")
    p.add_argument("--sensors", type=int, default=3, help="number of COMINT sensors")
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--duration", type=float, default=0, help="seconds to run (0 = forever)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", default=os.getenv("SIM_FRAMES", "data/sim_frames.npz"), help="npz bank of real frames (optional)")
    a = p.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    rng = np.random.default_rng(a.seed)
    frames = FrameSource(a.frames, rng)
    client = httpx.Client(timeout=10, trust_env=False)
    t_start, tick, sent, failed = time.time(), 0, 0, 0
    comint = [e for e in SCENARIO if e["mod"] != "pulse"]
    while not a.duration or time.time() - t_start < a.duration:
        drift = np.sin(tick / 60.0)  # slow bearing wander
        for s in range(a.sensors):
            e = comint[(tick + s) % len(comint)]
            x = frames(e["mod"], a.length, e["snr"] + rng.normal(0, 2))
            body = dict(sensorId=f"sirius-{s}", rfMhz=e["rf"], aoa=float((e["aoa"] + 3 * drift + rng.normal(0, 1.0)) % 360),
                        i=x[:, 0].tolist(), q=x[:, 1].tolist())
            try:
                r = client.post(f"{a.gateway}/ingest/iq", json=body)
                sent += r.status_code == 202
                failed += r.status_code != 202
            except httpx.HTTPError as err:
                failed += 1
                log.warning("gateway unreachable: %s", err)
                time.sleep(2)
        if tick % 2 == 0:
            try:
                r = client.post(f"{a.gateway}/ingest/pdw", json=dict(sensorId="sirius-esm", pdw=pdw_batch(rng, tick * 1000 / a.rate)))
                sent += r.status_code == 202
                failed += r.status_code != 202
            except httpx.HTTPError as err:
                failed += 1
                log.warning("gateway unreachable: %s", err)
        tick += 1
        if tick % 50 == 0:
            log.info("tick=%d sent=%d failed=%d", tick, sent, failed)
        time.sleep(max(0.0, t_start + tick / a.rate - time.time()))
    print(dict(ticks=tick, sent=sent, failed=failed, seconds=round(time.time() - t_start, 1)))


if __name__ == "__main__":
    main()

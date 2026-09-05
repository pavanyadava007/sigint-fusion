"""Async load test of POST /classify. Writes results/benchmark_classify.json (p50/p95/p99 + req/s, with host/GPU provenance).

python benchmark/load.py --url http://localhost:8000 --n 2000 --conc 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from datetime import UTC, datetime

import httpx
import numpy as np


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--conc", type=int, default=32)
    p.add_argument("--len", type=int, default=1024)
    p.add_argument("--out", default="results/benchmark_classify.json")
    p.add_argument("--label", default="", help="free-text description of the serving host, e.g. 'docker CPU, 4 vCPU'")
    a = p.parse_args()
    body = dict(i=np.random.randn(a.len).tolist(), q=np.random.randn(a.len).tolist())
    lat, model_lat, errors = [], [], 0
    sem = asyncio.Semaphore(a.conc)
    async with httpx.AsyncClient(timeout=30, trust_env=False) as c:
        health = (await c.get(a.url + "/health")).json()

        async def one():
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                try:
                    r = await c.post(a.url + "/classify", json=body)
                    r.raise_for_status()
                    model_lat.append(r.json()["latency_ms"])
                    lat.append(time.perf_counter() - t)
                except Exception:  # noqa: BLE001
                    errors += 1

        for _ in range(50):  # warm-up
            await one()
        lat.clear()
        model_lat.clear()
        t0 = time.perf_counter()
        await asyncio.gather(*[one() for _ in range(a.n)])
        wall = time.perf_counter() - t0
    end2end = np.array(lat) * 1e3
    res = dict(n=a.n, conc=a.conc, length=a.len, errors=errors, req_per_s=round(len(lat) / wall, 1),
               end_to_end_ms={k: round(float(np.percentile(end2end, q)), 2) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))},
               model_only_ms={k: round(float(np.percentile(model_lat, q)), 3) for k, q in (("p50", 50), ("p95", 95), ("p99", 99))},
               providers=health.get("providers"), model=health.get("model"), label=a.label,
               client_host=platform.node(), cpu=platform.processor() or platform.machine(), url=a.url,
               timestamp=datetime.now(UTC).isoformat(timespec="seconds"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    asyncio.run(main())

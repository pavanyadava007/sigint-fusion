"""Synthetic radar PDW generator for R-ESM experiments (ground truth included)."""

from __future__ import annotations

import numpy as np


def synth_pdw(n_emitters: int = 4, duration_ms: float = 50, seed: int = 0, aoa_noise: float = 1.5, rf_noise: float = 0.5):
    """Returns (pdw [N,4] = TOA ms, RF MHz, PW us, AOA deg; true emitter id [N]; list of emitter truth dicts)."""
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for e in range(n_emitters):
        rf, pw, aoa = rng.uniform(2000, 18000), rng.uniform(0.2, 20), rng.uniform(0, 360)
        kind = str(rng.choice(["constant", "stagger", "jitter", "agile"]))
        base_pri = rng.uniform(0.1, 2.0)  # ms
        t, k = rng.uniform(0, base_pri), 0
        while t < duration_ms:
            pri = {"constant": base_pri, "stagger": base_pri * (1 + 0.3 * (k % 3)), "jitter": base_pri * rng.uniform(0.85, 1.15), "agile": base_pri}[kind]
            rf_k = rf + (rng.choice([-200, 0, 200]) if kind == "agile" else 0)
            rows.append([t, rf_k + rng.normal(0, rf_noise), pw * rng.uniform(0.98, 1.02), (aoa + rng.normal(0, aoa_noise)) % 360, e])
            t += pri
            k += 1
        truth.append(dict(id=e, rf=rf, pw=pw, aoa=aoa, pri=base_pri, type=kind))
    rows = np.array(sorted(rows))
    return rows[:, :4], rows[:, 4].astype(int), truth


def purity_completeness(pred, true) -> dict:
    from sklearn.metrics import completeness_score, homogeneity_score

    return dict(purity=float(homogeneity_score(true, pred)), completeness=float(completeness_score(true, pred)))

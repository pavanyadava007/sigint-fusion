"""Specific Emitter Identification data.

(1) synth_sei: imprint per-device hardware fingerprints (I/Q imbalance, DC offset, CFO, phase noise, PA nonlinearity) on clean I/Q.
(2) load_oracle: ORACLE dataset (16 USRP X310 emitters, genesys-lab.org) if present; one .sigmf-data file per device/distance.
"""

from __future__ import annotations

import glob
import os

import numpy as np


def _fingerprint(rng: np.random.Generator) -> dict:
    return dict(iq_gain=rng.normal(1.0, 0.05), iq_phase=rng.normal(0, 0.06), dc=rng.normal(0, 0.02, 2),
                cfo=rng.normal(0, 0.004), pn=rng.uniform(0.002, 0.02), pa=rng.uniform(0.02, 0.15))


def impair(x: np.ndarray, fp: dict, rng: np.random.Generator) -> np.ndarray:
    """x:[2,L] float. Applies device fingerprint fp. Returns [2,L] float32."""
    L = x.shape[1]
    z = x[0] + 1j * x[1]
    z = z * np.exp(1j * 2 * np.pi * fp["cfo"] * np.arange(L))  # carrier frequency offset
    z = z * np.exp(1j * np.cumsum(rng.normal(0, fp["pn"], L)))  # phase noise (Wiener)
    z = z - fp["pa"] * np.abs(z) ** 2 * z  # 3rd-order PA nonlinearity
    i = fp["iq_gain"] * z.real + fp["dc"][0]
    q = z.imag * np.cos(fp["iq_phase"]) + z.real * np.sin(fp["iq_phase"]) + fp["dc"][1]
    return np.stack([i, q]).astype(np.float32)


def synth_sei(X: np.ndarray, n_devices: int = 16, per_device: int = 400, seed: int = 0):
    """X: clean I/Q [N,2,L]. Returns (Xsei [n_devices*per_device,2,L], device_id, fingerprints)."""
    rng = np.random.default_rng(seed)
    fps = [_fingerprint(rng) for _ in range(n_devices)]
    idx = rng.choice(len(X), n_devices * per_device)
    out, y = [], []
    for d in range(n_devices):
        for i in idx[d * per_device : (d + 1) * per_device]:
            out.append(impair(X[i], fps[d], rng))
            y.append(d)
    out = np.stack(out)
    p = np.sqrt((out**2).sum(axis=(1, 2), keepdims=True) / out.shape[2]) + 1e-8
    return (out / p).astype(np.float32), np.array(y), fps


def load_oracle(root: str, L: int = 1024, per_file: int = 2000):
    """ORACLE: one .sigmf-data (complex64) file per device. Returns X[N,2,L], y."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.sigmf-data"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .sigmf-data files under {root}")
    X, y = [], []
    for d, f in enumerate(files):
        z = np.fromfile(f, np.complex64)[: per_file * L].reshape(-1, L)
        X.append(np.stack([z.real, z.imag], 1))
        y += [d] * len(z)
    X = np.concatenate(X)
    p = np.sqrt((X**2).sum(axis=(1, 2), keepdims=True) / L) + 1e-8
    return (X / p).astype(np.float32), np.array(y)

"""Synthetic modulation dataset generator (RadioML-compatible layouts).

RadioML 2016.10a / 2018.01A need a DeepSig licence, so this module synthesises the same 24 modulation classes
with a baseband channel model (RRC pulse shaping, CFO, phase offset, sample-rate offset, multipath, AWGN) and
writes files in the SAME on-disk layout, so `data.radioml.load_2016/load_2018` and every training script run
unchanged. Every number produced from these files must be labelled "synthetic" in reports.

    python -m data.synth_mod --out data/synth_2018.hdf5 --length 1024 --per-class 4000
    python -m data.synth_mod --out data/synth_2016.pkl  --length 128  --per-class 1000 --format 2016
"""

from __future__ import annotations

import argparse
import pickle
import time
from multiprocessing import Pool

import h5py
import numpy as np
from scipy.signal import fftconvolve, firwin, hilbert

from data.radioml import MODS_2018

SPS = 8  # samples per symbol
RRC_BETA = 0.35
SNR_RANGE = (-20, 30)  # dB, 2 dB steps like RadioML 2018


# --------------------------------------------------------------------------- constellations
def _psk(m: int) -> np.ndarray:
    return np.exp(1j * (2 * np.pi * np.arange(m) / m + np.pi / m))


def _square_qam(m: int) -> np.ndarray:
    k = int(np.sqrt(m))
    lv = np.arange(-(k - 1), k, 2)
    return (lv[None, :] + 1j * lv[:, None]).ravel()


def _cross_qam(m: int) -> np.ndarray:
    """32-QAM (6x6 minus corners) and 128-QAM (12x12 minus 2x2 corners)."""
    k, cut = {32: (6, 1), 128: (12, 2)}[m]
    lv = np.arange(-(k - 1), k, 2)
    pts = [complex(a, b) for a in lv for b in lv if not (abs(a) > lv[-1] - 2 * cut and abs(b) > lv[-1] - 2 * cut)]
    assert len(pts) == m, (m, len(pts))
    return np.array(pts)


def _apsk(rings: list[int], radii: list[float]) -> np.ndarray:
    pts = []
    for n, r in zip(rings, radii):
        pts += list(r * np.exp(1j * (2 * np.pi * np.arange(n) / n + np.pi / n)))
    return np.array(pts)


CONSTELLATIONS: dict[str, np.ndarray] = {
    "OOK": np.array([0.0, 1.0]),
    "4ASK": np.array([-3.0, -1.0, 1.0, 3.0]),
    "8ASK": np.arange(-7.0, 8.0, 2.0),
    "BPSK": np.array([-1.0, 1.0]),
    "QPSK": _psk(4),
    "8PSK": _psk(8),
    "16PSK": _psk(16),
    "32PSK": _psk(32),
    "16APSK": _apsk([4, 12], [1.0, 2.57]),
    "32APSK": _apsk([4, 12, 16], [1.0, 2.53, 4.30]),
    "64APSK": _apsk([4, 12, 20, 28], [1.0, 2.2, 3.6, 5.2]),
    "128APSK": _apsk([4, 12, 20, 28, 64], [1.0, 2.2, 3.6, 5.2, 7.0]),
    "16QAM": _square_qam(16),
    "32QAM": _cross_qam(32),
    "64QAM": _square_qam(64),
    "128QAM": _cross_qam(128),
    "256QAM": _square_qam(256),
}
ANALOG = {"AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM"}
assert set(CONSTELLATIONS) | ANALOG | {"GMSK", "OQPSK"} == set(MODS_2018)


# --------------------------------------------------------------------------- filters
def rrc_taps(beta: float = RRC_BETA, sps: int = SPS, span: int = 8) -> np.ndarray:
    n = span * sps
    t = (np.arange(-n // 2, n // 2 + 1)) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            taps[i] = 1 + beta * (4 / np.pi - 1)
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            taps[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                np.pi * ti * (1 - (4 * beta * ti) ** 2)
            )
    return taps / np.sqrt((taps**2).sum())


_RRC = rrc_taps()
_GAUSS = None
_LPF = firwin(63, 0.05)  # analog message band-limit (fraction of Nyquist)


def _gauss_taps(bt: float = 0.3, sps: int = SPS, span: int = 4) -> np.ndarray:
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    a = np.sqrt(np.log(2) / 2) / bt
    g = (np.sqrt(np.pi) / a) * np.exp(-((np.pi * t / a) ** 2))
    return g / g.sum()


# --------------------------------------------------------------------------- sources
def _message(rng: np.random.Generator, n: int) -> np.ndarray:
    """Band-limited analog message with unit peak, like speech/music envelope."""
    m = fftconvolve(rng.standard_normal(n + 200), _LPF, mode="same")[100 : 100 + n]
    return m / (np.abs(m).max() + 1e-9)


def _linear(rng: np.random.Generator, const: np.ndarray, n: int, offset_q: bool = False) -> np.ndarray:
    nsym = n // SPS + 16
    sym = const[rng.integers(len(const), size=nsym)].astype(np.complex128)
    up = np.zeros(nsym * SPS, np.complex128)
    up[::SPS] = sym
    if offset_q:  # OQPSK: delay quadrature by half a symbol
        up = up.real + 1j * np.roll(up.imag, SPS // 2)
    x = fftconvolve(up, _RRC, mode="same")
    start = rng.integers(0, SPS)
    return x[start + 4 * SPS : start + 4 * SPS + n]


def _gmsk(rng: np.random.Generator, n: int) -> np.ndarray:
    global _GAUSS
    if _GAUSS is None:
        _GAUSS = _gauss_taps()
    nsym = n // SPS + 16
    bits = rng.integers(2, size=nsym) * 2 - 1
    up = np.repeat(bits, SPS).astype(np.float64)
    freq = fftconvolve(up, _GAUSS, mode="same")
    phase = np.cumsum(freq) * (np.pi * 0.5 / SPS)  # modulation index h = 0.5
    x = np.exp(1j * phase)
    start = rng.integers(0, SPS)
    return x[start + 4 * SPS : start + 4 * SPS + n]


def _analog(rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
    m = _message(rng, n)
    if kind == "FM":
        return np.exp(1j * 2 * np.pi * 0.08 * np.cumsum(m))
    if kind == "AM-DSB-WC":
        return (1 + 0.8 * m).astype(np.complex128)
    if kind == "AM-DSB-SC":
        return m.astype(np.complex128)
    an = hilbert(m)  # analytic signal keeps the upper sideband
    if kind == "AM-SSB-SC":
        return an
    return 1 + 0.8 * an  # AM-SSB-WC


def baseband(rng: np.random.Generator, mod: str, n: int) -> np.ndarray:
    if mod in CONSTELLATIONS:
        return _linear(rng, CONSTELLATIONS[mod], n)
    if mod == "OQPSK":
        return _linear(rng, _psk(4), n, offset_q=True)
    if mod == "GMSK":
        return _gmsk(rng, n)
    return _analog(rng, mod, n)


# --------------------------------------------------------------------------- channel
def channel(rng: np.random.Generator, x: np.ndarray, snr_db: float) -> np.ndarray:
    n = len(x)
    # sample-rate offset via fractional resampling (+/- 50 ppm .. 500 ppm)
    sro = rng.uniform(-5e-4, 5e-4)
    t = np.arange(n) * (1 + sro)
    x = np.interp(t, np.arange(n), x.real) + 1j * np.interp(t, np.arange(n), x.imag)
    # sparse multipath: 1-3 taps with random delay/gain
    h = np.zeros(6, np.complex128)
    h[0] = 1.0
    for _ in range(rng.integers(0, 3)):
        h[rng.integers(1, 6)] += rng.uniform(0.05, 0.4) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    x = fftconvolve(x, h, mode="full")[:n]
    # CFO + phase
    cfo = rng.uniform(-0.01, 0.01)
    x = x * np.exp(1j * (2 * np.pi * cfo * np.arange(n) + rng.uniform(0, 2 * np.pi)))
    # AWGN at target SNR
    p = np.mean(np.abs(x) ** 2) + 1e-12
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(p / (2 * 10 ** (snr_db / 10)))
    return x + noise


def make_sample(mod: str, n: int, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = channel(rng, baseband(rng, mod, n), snr_db)
    x = x / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-9)
    return np.stack([x.real, x.imag], 1).astype(np.float32)  # [n, 2]  (RadioML 2018 layout)


def _worker(args):
    mods, n, snrs, seeds = args
    return np.stack([make_sample(m, n, s, sd) for m, s, sd in zip(mods, snrs, seeds)])


def generate(per_class: int, length: int, seed: int = 0, workers: int = 8, mods: list[str] | None = None):
    """Returns X[N,length,2] float32, y[N] int, snr[N] int, mods."""
    mods = mods or MODS_2018
    rng = np.random.default_rng(seed)
    snr_grid = np.arange(SNR_RANGE[0], SNR_RANGE[1] + 1, 2)
    y = np.repeat(np.arange(len(mods)), per_class)
    snr = rng.choice(snr_grid, size=len(y))
    seeds = rng.integers(0, 2**31 - 1, size=len(y))
    names = [mods[i] for i in y]
    chunks = [(names[i : i + 256], length, snr[i : i + 256], seeds[i : i + 256]) for i in range(0, len(y), 256)]
    with Pool(workers) as pool:
        parts = pool.map(_worker, chunks)
    return np.concatenate(parts), y, snr.astype(np.int64), mods


def write_2018(path: str, X, y, snr, mods):
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=1)
        f.create_dataset("Y", data=np.eye(len(mods), dtype=np.int64)[y])
        f.create_dataset("Z", data=snr[:, None])
        f.attrs["classes"] = list(mods)
        f.attrs["synthetic"] = True


def write_2016(path: str, X, y, snr, mods):
    d = {}
    for c, m in enumerate(mods):
        for s in np.unique(snr):
            sel = (y == c) & (snr == s)
            if sel.any():
                d[(m, int(s))] = X[sel].transpose(0, 2, 1).astype(np.float32)  # [N,2,L] (2016 layout)
    pickle.dump(d, open(path, "wb"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--per-class", type=int, default=4000)
    p.add_argument("--format", choices=["2018", "2016"], default="2018")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    t0 = time.time()
    X, y, snr, mods = generate(a.per_class, a.length, a.seed, a.workers)
    (write_2018 if a.format == "2018" else write_2016)(a.out, X, y, snr, mods)
    print(dict(out=a.out, n=len(y), length=a.length, classes=len(mods), seconds=round(time.time() - t0, 1)))


if __name__ == "__main__":
    main()

"""Classical SIGINT signal processing: PSD, spectrogram, CA-CFAR detection, PDW deinterleaving."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft, welch
from sklearn.cluster import DBSCAN


def psd(iq: np.ndarray, fs: float = 1.0, nperseg: int = 256):
    """Welch PSD of complex baseband iq[2,L]. Returns (freqs, dB) centred on 0 Hz."""
    f, P = welch(iq[0] + 1j * iq[1], fs=fs, nperseg=min(nperseg, iq.shape[1]), return_onesided=False)
    return np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(P) + 1e-12)


def spectrogram(iq: np.ndarray, fs: float = 1.0, nperseg: int = 64):
    """STFT magnitude in dB, shape [freq, time], centred on 0 Hz."""
    f, t, Z = stft(iq[0] + 1j * iq[1], fs=fs, nperseg=nperseg, return_onesided=False)
    return np.fft.fftshift(f), t, 20 * np.log10(np.abs(np.fft.fftshift(Z, axes=0)) + 1e-12)


def ca_cfar(x_db: np.ndarray, guard: int = 2, train: int = 8, pfa: float = 1e-3) -> np.ndarray:
    """1-D cell-averaging CFAR on a power vector (dB). Vectorised; edges use the cells that exist.

    Threshold factor alpha = N (Pfa^(-1/N) - 1) for N training cells (Richards, Fundamentals of Radar SP).
    """
    lin = 10 ** (np.asarray(x_db, float) / 10)
    n = len(lin)
    ntrain = 2 * train
    alpha = ntrain * (pfa ** (-1 / ntrain) - 1)
    cs = np.concatenate([[0.0], np.cumsum(lin)])
    i = np.arange(n)
    lo = np.clip(i - guard - train, 0, n)
    lo_end = np.clip(i - guard, 0, n)
    hi_start = np.clip(i + guard + 1, 0, n)
    hi = np.clip(i + guard + train + 1, 0, n)
    total = (cs[lo_end] - cs[lo]) + (cs[hi] - cs[hi_start])
    count = (lo_end - lo) + (hi - hi_start)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.inf)
    return lin > alpha * mean


def deinterleave_pdw(pdw: np.ndarray, eps: float = 0.15, min_samples: int = 5):
    """pdw: [N,4] columns TOA(ms), RF(MHz), PW(us), AOA(deg). DBSCAN in standardised (RF,PW,AOA) space.

    Returns (labels[N], emitters{label: stats}). PRI type: 'constant' (CV<5%), 'stagger' (few repeating levels),
    'jitter' (continuous spread), 'agile' when RF hops between discrete values.
    """
    pdw = np.asarray(pdw, float)
    if len(pdw) == 0:
        return np.zeros(0, int), {}
    feats = pdw[:, 1:4] / (pdw[:, 1:4].std(0) + 1e-9)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(feats)
    emitters = {}
    for lab in sorted(set(labels) - {-1}):
        m = labels == lab
        toa = np.sort(pdw[m, 0])
        pri = np.diff(toa)
        rf = pdw[m, 1]
        stats = dict(n=int(m.sum()), rf=float(rf.mean()), pw=float(pdw[m, 2].mean()), aoa=float(pdw[m, 3].mean()),
                     pri_mean=float(pri.mean()) if len(pri) else None, pri_type=_pri_type(pri, rf))
        emitters[int(lab)] = stats
    return labels, emitters


def _pri_type(pri: np.ndarray, rf: np.ndarray) -> str:
    if len(pri) < 3:
        return "unknown"
    if len(np.unique(np.round(rf / 50))) >= 3:  # RF spread across several 50 MHz bins
        return "agile"
    cv = pri.std() / (pri.mean() + 1e-9)
    if cv < 0.05:
        return "constant"
    levels = len(np.unique(np.round(pri / (pri.mean() * 0.05))))
    return "stagger" if levels <= 6 else "jitter"

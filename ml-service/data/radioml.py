"""RadioML loaders (also read the synthetic files written by `data.synth_mod`, which use the same layouts). No torch dependency;
the torch Dataset lives in `data.dataset`.

2016.10a: pickle dict {(mod, snr): [N,2,128]}.
2018.01A: HDF5 X[N,1024,2] float32, Y one-hot [N,24], Z snr [N,1]. The real file is ~21 GB, so `load_2018` reads
it in chunks and filters on the fly instead of materialising the whole array.
"""

from __future__ import annotations

import pickle

import h5py
import numpy as np

# Label order of the DeepSig 2018.01A one-hot vectors = the order in the RadioML 2018 paper (O'Shea et al. 2018). The classes.txt
# shipped inside the tarball lists the classes in a DIFFERENT order and is NOT the label order: scripts/check_2018_labels.py verifies
# this from signal physics (OOK has the largest envelope variation, OOK/ASK/BPSK show a squared-signal spectral line, FM/GMSK are
# constant-envelope), and every signature matches this order, none match classes.txt. Synthetic files carry their own `classes` attr.
MODS_2018 = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK", "16APSK", "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
]  # fmt: skip


def class_names_2018(path: str, f: h5py.File | None = None) -> list[str]:
    """Label order used by a 2018-layout file: the HDF5 `classes` attribute when present (synthetic files), else MODS_2018."""
    if f is not None and "classes" in f.attrs:
        return [str(c) for c in f.attrs["classes"]]
    return list(MODS_2018)


def normalise(x: np.ndarray) -> np.ndarray:
    """Per-sample RMS power normalisation, x:[N,2,L]."""
    p = np.sqrt((x**2).sum(axis=(1, 2), keepdims=True) / x.shape[2]) + 1e-8
    return (x / p).astype(np.float32)


def load_2016(path: str, snr_min: int = -20):
    d = pickle.load(open(path, "rb"), encoding="latin1")
    mods = sorted({k[0] for k in d})
    X, y, snr = [], [], []
    for (m, s), v in d.items():
        if s < snr_min:
            continue
        X.append(v)
        y += [mods.index(m)] * len(v)
        snr += [s] * len(v)
    return normalise(np.concatenate(X)), np.array(y), np.array(snr), mods


def load_2018(path: str, classes: list[str] | None = None, snr_min: int = -20, max_per_class: int | None = None,
              seed: int = 0, chunk: int = 65536):
    """classes: modulation names to keep (held-out few-shot splits). Reads the file chunk-wise (memory-safe on the 21 GB original)."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        names = class_names_2018(path, f)
        keep_idx = None if classes is None else np.array([names.index(c) for c in classes])
        Y = f["Y"][:].argmax(1)
        Z = f["Z"][:, 0].astype(np.int64)
        keep = Z >= snr_min
        if keep_idx is not None:
            keep &= np.isin(Y, keep_idx)
        sel = np.where(keep)[0]
        if max_per_class:
            parts = [rng.choice(ci, min(max_per_class, len(ci)), replace=False) for ci in
                     (sel[Y[sel] == c] for c in np.unique(Y[sel]))]
            sel = np.sort(np.concatenate(parts))
        Xs = []
        for lo in range(0, len(Y), chunk):  # contiguous reads, then fancy-index in memory
            hi = min(lo + chunk, len(Y))
            s = sel[(sel >= lo) & (sel < hi)]
            if len(s):
                Xs.append(f["X"][lo:hi][s - lo])
        X = np.concatenate(Xs) if Xs else np.zeros((0, f["X"].shape[1], 2), np.float32)
    y, Z = Y[sel], Z[sel]
    if keep_idx is not None:
        remap = {c: i for i, c in enumerate(keep_idx)}
        y = np.array([remap[v] for v in y])
    return normalise(X.transpose(0, 2, 1)), y, Z, (classes or names)

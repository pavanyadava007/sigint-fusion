"""Verify the label order of a RadioML 2018.01A HDF5 from signal physics (no labels trusted).

For each label index, on the 30 dB frames: envelope coefficient of variation (OOK largest, FM/GMSK ~0), the squared-signal
spectral line (strong only for real-valued constellations OOK/ASK/BPSK), and the 4th-power line (QPSK/OQPSK). Prints the
signatures next to the names from `data.radioml.MODS_2018` and flags impossibilities. python scripts/check_2018_labels.py data/GOLD_XYZ_OSC.0001_1024.hdf5
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml-service"))
from data.radioml import class_names_2018  # noqa: E402


def main(path: str, n: int = 64):
    with h5py.File(path) as f:
        names = class_names_2018(path, f)
        Y, Z = f["Y"][:].argmax(1), f["Z"][:, 0]
        hi = np.where(Z == Z.max())[0]
        rows = []
        for c in range(len(names)):
            idx = np.sort(hi[Y[hi] == c][:n])
            x = f["X"][idx].astype(np.float32)
            z = x[..., 0] + 1j * x[..., 1]
            env = np.abs(z)
            cv = float(np.mean(env.std(1) / env.mean(1)))
            l2 = np.abs(np.fft.fft(z**2, axis=1))
            l4 = np.abs(np.fft.fft(z**4, axis=1))
            rows.append((c, names[c], cv, float(np.mean(l2.max(1) / l2.mean(1))), float(np.mean(l4.max(1) / l4.mean(1)))))
    ok = True
    by = {r[1]: r for r in rows}
    for c, name, cv, line2, line4 in rows:
        print(f"idx {c:2d} {name:10s} envelope-CV {cv:.3f}  squared-line {line2:6.1f}  4th-power-line {line4:6.1f}")
    checks = [("OOK has the largest envelope variation", by["OOK"][2] == max(r[2] for r in rows)),
              ("FM is constant-envelope (CV < 0.05)", by["FM"][2] < 0.05),
              ("GMSK is near constant-envelope (CV < 0.15)", by["GMSK"][2] < 0.15),
              ("OOK/4ASK/8ASK/BPSK show a squared-signal line (> 40)", all(by[k][3] > 40 for k in ("OOK", "4ASK", "8ASK", "BPSK"))),
              ("QPSK/8PSK/16QAM show no squared-signal line (< 30)", all(by[k][3] < 30 for k in ("QPSK", "8PSK", "16QAM")))]
    for text, passed in checks:
        ok &= passed
        print(("PASS " if passed else "FAIL ") + text)
    print("label order consistent with data.radioml.MODS_2018" if ok else "LABEL ORDER MISMATCH: fix MODS_2018 before training")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ml-service/data/GOLD_XYZ_OSC.0001_1024.hdf5")

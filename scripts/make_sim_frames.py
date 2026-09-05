"""Extract a small bank of REAL RadioML 2018.01A frames for the sensor simulator (data/sim_frames.npz, a few MB).

The simulator streams these instead of synthetic frames when the file exists, so the live demo classifies real signals with
the real-data model. python scripts/make_sim_frames.py --data ml-service/data/GOLD_XYZ_OSC.0001_1024.hdf5 --per-class 300
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml-service"))
from data.radioml import MODS_2018, load_2018  # noqa: E402

MODS = ["FM", "AM-DSB-WC", "OOK", "QPSK", "GMSK", "BPSK", "16QAM", "8PSK"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ml-service/data/GOLD_XYZ_OSC.0001_1024.hdf5")
    p.add_argument("--out", default="ml-service/data/sim_frames.npz")
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--snr-min", type=int, default=6)
    a = p.parse_args()
    mods = [m for m in MODS if m in MODS_2018]
    X, y, snr, classes = load_2018(a.data, classes=mods, snr_min=a.snr_min, max_per_class=a.per_class, seed=1)
    np.savez_compressed(a.out, X=X.astype(np.float16), y=y, snr=snr, classes=np.array(classes), source=os.path.basename(a.data))
    print(dict(out=a.out, frames=int(len(y)), classes=classes, snr_min=a.snr_min, mb=round(os.path.getsize(a.out) / 1e6, 1)))


if __name__ == "__main__":
    main()

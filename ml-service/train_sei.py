"""Specific Emitter Identification: 16 devices (synthetic hardware fingerprints on QPSK, or ORACLE captures).
Supervised on 12 devices, then 5-shot prototypical evaluation on 4 UNSEEN devices.

python train_sei.py --data data/synth_2018.hdf5 --init ckpt/ft.pt          # synthetic fingerprints
python train_sei.py --oracle data/oracle --init ckpt/ft.pt                  # real ORACLE captures if available
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from train import dev, fit, load_backbone, provenance, seed_all

from data.dataset import IQDataset
from data.radioml import load_2018
from data.sei import load_oracle, synth_sei
from models.resnet1d import ResNet1D


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data")
    p.add_argument("--oracle")
    p.add_argument("--init")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--out", default="ckpt/sei.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="sei")
    a = p.parse_args()
    seed_all(a.seed)
    t0 = time.time()
    if a.oracle:
        X, y = load_oracle(a.oracle)
        source = "oracle"
    else:
        Xc, *_ = load_2018(a.data, classes=["QPSK"], snr_min=18, max_per_class=8000, seed=a.seed)
        X, y, _ = synth_sei(Xc, seed=a.seed)
        source = "synthetic_fingerprints"
    n = int(y.max() + 1)
    base, novel = list(range(n - 4)), list(range(n - 4, n))
    mb, mn = np.isin(y, base), np.isin(y, novel)
    model = ResNet1D(len(base)).to(dev)
    if a.init:
        load_backbone(model, a.init)
    Xb, yb = X[mb], y[mb]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(yb))
    s = int(0.8 * len(yb))
    best, history = fit(model, Xb[perm[:s]], yb[perm[:s]], Xb[perm[s:]], yb[perm[s:]], a.epochs, 1e-3, 256, lambda e, d: print(e, d, flush=True))
    Xn, yn = X[mn], y[mn] - novel[0]
    model.eval()
    with torch.no_grad():
        E_all = torch.cat([model.embed(x.to(dev)) for x, _ in DataLoader(IQDataset(Xn, yn), 512)])
    accs = []
    for _ in range(a.episodes):
        sup, protos = [], []
        for c in range(4):
            si = rng.choice(np.where(yn == c)[0], a.shots, replace=False)
            sup += list(si)
            protos.append(E_all[si].mean(0))
        q = np.setdiff1d(np.arange(len(yn)), sup)
        pred = torch.cdist(E_all[q], torch.stack(protos)).argmin(1).cpu().numpy()
        accs.append(float((pred == yn[q]).mean()))
    a.data = a.oracle or a.data
    result = dict(mode="sei", name=a.name, source=source, n_devices=n, base_devices=len(base), novel_devices=4, init=a.init,
                  base_val_acc=round(best, 4), history=history,
                  fewshot_novel=dict(shots=a.shots, episodes=a.episodes, acc_mean=round(float(np.mean(accs)), 4),
                                     acc_std=round(float(np.std(accs)), 4), chance=0.25),
                  seconds=round(time.time() - t0, 1), **provenance(a))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(dict(model=model.state_dict(), arch="resnet", classes=[f"device-{d}" for d in base], length=int(X.shape[2])), a.out)
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(f"results/{a.name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ("history", "args")}), flush=True)


if __name__ == "__main__":
    main()

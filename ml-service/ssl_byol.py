"""BYOL self-supervised pretraining on UNLABELLED I/Q (labels are ignored). Output: backbone checkpoint for `train.py --init`.

python ssl_byol.py --data data/synth_2018.hdf5 --epochs 20 --out ckpt/byol.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from train import provenance, seed_all

from data.radioml import load_2018
from models.resnet1d import ResNet1D

dev = "cuda" if torch.cuda.is_available() else "cpu"


def augment(x: torch.Tensor) -> torch.Tensor:
    """RF-plausible views: random phase rotation, CFO, SNR jitter, circular time shift (per sample)."""
    B, _, L = x.shape
    th = torch.rand(B, 1, device=x.device) * 2 * np.pi
    c, s = torch.cos(th), torch.sin(th)
    i, q = x[:, 0], x[:, 1]
    x = torch.stack([c * i - s * q, s * i + c * q], 1)
    cfo = (torch.rand(B, 1, device=x.device) - 0.5) * 0.01
    ph = 2 * np.pi * cfo * torch.arange(L, device=x.device)
    z = torch.complex(x[:, 0], x[:, 1]) * torch.exp(1j * ph)
    x = torch.stack([z.real, z.imag], 1)
    x = x + torch.randn_like(x) * torch.rand(B, 1, 1, device=x.device) * 0.3
    shifts = torch.randint(0, L, (B,), device=x.device)
    idx = (torch.arange(L, device=x.device)[None, :] - shifts[:, None]) % L
    return torch.gather(x, 2, idx[:, None, :].expand(-1, 2, -1))


class Unlabeled(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i])


def mlp(i, h=512, o=128):
    return nn.Sequential(nn.Linear(i, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Linear(h, o))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--out", default="ckpt/byol.pt")
    p.add_argument("--tau", type=float, default=0.99)
    p.add_argument("--max_per_class", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="byol")
    a = p.parse_args()
    seed_all(a.seed)
    t0 = time.time()
    X, *_ = load_2018(a.data, snr_min=0, max_per_class=a.max_per_class, seed=a.seed)
    online, proj, pred = ResNet1D(1).to(dev), mlp(128).to(dev), mlp(128).to(dev)
    target, tproj = copy.deepcopy(online), copy.deepcopy(proj)
    for q in list(target.parameters()) + list(tproj.parameters()):
        q.requires_grad = False
    opt = torch.optim.AdamW(list(online.parameters()) + list(proj.parameters()) + list(pred.parameters()), 1e-3)
    dl = DataLoader(Unlabeled(X), a.bs, shuffle=True, drop_last=True, num_workers=4, pin_memory=dev == "cuda")
    history = []
    for e in range(a.epochs):
        tot = 0.0
        for x in dl:
            x = x.to(dev, non_blocking=True)
            v1, v2 = augment(x), augment(x)

            def loss_fn(a_, b_):
                p_ = nn.functional.normalize(pred(proj(online.embed(a_))), dim=1)
                with torch.no_grad():
                    t_ = nn.functional.normalize(tproj(target.embed(b_)), dim=1)
                return 2 - 2 * (p_ * t_).sum(1).mean()

            loss = loss_fn(v1, v2) + loss_fn(v2, v1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                for o, t in zip(list(online.parameters()) + list(proj.parameters()), list(target.parameters()) + list(tproj.parameters())):
                    t.mul_(a.tau).add_((1 - a.tau) * o)
            tot += loss.item()
        history.append(dict(epoch=e, loss=round(tot / len(dl), 4)))
        print(history[-1], flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(dict(model=online.state_dict(), arch="resnet", classes=None, length=int(X.shape[2])), a.out)
    os.makedirs("results", exist_ok=True)
    result = dict(mode="byol", name=a.name, n_unlabeled=int(len(X)), epochs=a.epochs, history=history, out=a.out,
                  seconds=round(time.time() - t0, 1), **provenance(a))
    json.dump(result, open(f"results/{a.name}.json", "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()

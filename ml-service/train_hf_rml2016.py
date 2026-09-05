"""Synthetic-to-real transfer check on REAL RadioML 2016.10a frames (community Hugging Face mirror hitrs909/RML2016, config 6db_0:
11 classes, 128 samples, single SNR slice, 9900/539/1100 train/val/test). The mirror does not publish class names, only label ids.

Compares, over several seeds, on the held-out test split:
  scratch      ResNet-1D trained from random init on the real train split
  pretrained   backbone initialised from the synthetic-data checkpoint (--init), whole network fine-tuned
  linear-probe backbone from --init FROZEN, only the head trained (does synthetic pretraining produce usable features?)
at 100 % and 10 % of the real train split. Writes results/real_rml2016_6db.json with synthetic=False.

python train_hf_rml2016.py --data data/hf_rml2016 --init ckpt/pre.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from train import dev, fit, load_backbone, provenance, run_epoch, seed_all

from data.dataset import IQDataset
from data.radioml import normalise
from models.resnet1d import ResNet1D


def load_split(d: str, split: str):
    df = pq.read_table(os.path.join(d, f"{split}.parquet")).to_pandas()
    X = np.array([np.stack(v) for v in df["signal"]], np.float32)  # [N,2,128]
    X = X - X.mean(axis=2, keepdims=True)  # the mirror offset the samples to ~0.5; remove the DC term per channel
    return normalise(X), df["label_id"].to_numpy().astype(np.int64)


@torch.no_grad()
def test_acc(model, X, y):
    _, acc = run_epoch(model, DataLoader(IQDataset(X, y), 512))
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/hf_rml2016")
    p.add_argument("--init", default="ckpt/pre.pt", help="synthetic-data checkpoint whose backbone is transferred")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--name", default="real_rml2016_6db")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    Xtr, ytr = load_split(a.data, "train")
    Xva, yva = load_split(a.data, "val")
    Xte, yte = load_split(a.data, "test")
    n_cls = int(ytr.max() + 1)
    print(dict(train=len(ytr), val=len(yva), test=len(yte), classes=n_cls, device=dev), flush=True)
    runs: dict[str, dict[str, list[float]]] = {}
    for frac in (1.0, 0.1):
        for variant in ("scratch", "pretrained", "linear_probe"):
            accs = []
            for s in range(a.seeds):
                seed_all(s)
                Xs, ys = (Xtr, ytr) if frac == 1.0 else train_test_split(Xtr, ytr, train_size=frac, stratify=ytr, random_state=s)[0::2]
                model = ResNet1D(n_cls).to(dev)
                if variant != "scratch":
                    load_backbone(model, a.init)
                if variant == "linear_probe":
                    model.freeze_backbone(True)
                fit(model, Xs, ys, Xva, yva, a.epochs, 1e-3, 128, lambda e, d: None, workers=2)
                acc = test_acc(model, Xte, yte)
                accs.append(round(acc, 4))
                print(dict(frac=frac, variant=variant, seed=s, n_train=len(ys), test_acc=round(acc, 4)), flush=True)
            runs[f"{variant}@{frac}"] = dict(test_acc=accs, mean=round(float(np.mean(accs)), 4), std=round(float(np.std(accs)), 4), n_train=int(len(ys)))
    a.data = "hf:hitrs909/RML2016 (6db_0)"
    prov = provenance(a)
    prov["synthetic"] = False
    result = dict(mode="real_transfer", name=a.name, dataset="RadioML 2016.10a frames, single-SNR slice, Hugging Face mirror hitrs909/RML2016 config 6db_0",
                  n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)), classes=n_cls, class_names="not published by the mirror (label ids 0-10)",
                  length=int(Xtr.shape[2]), epochs=a.epochs, seeds=a.seeds, init=a.init, runs=runs, chance=round(1 / n_cls, 4),
                  seconds=round(time.time() - t0, 1), **prov)
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(f"results/{a.name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()

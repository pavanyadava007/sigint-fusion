"""Train / fine-tune / few-shot evaluate I/Q modulation classifiers.

Modes
  pretrain  : supervised on a 2016.10a-layout pickle (short frames)
  finetune  : 2018.01A-layout HDF5, 21 base classes; --init loads a backbone (head dropped); --frac for data-efficiency
  fewshot   : prototypical nearest-centroid on frozen embeddings for the 3 held-out classes (no training)

Every run writes results/<name>.json (metrics + provenance: dataset path, synthetic flag, GPU, seed, git commit).
Examples
  python train.py pretrain --data data/synth_2016.pkl --out ckpt/pre.pt
  python train.py finetune --data data/synth_2018.hdf5 --init ckpt/pre.pt --out ckpt/ft.pt
  python train.py finetune --data data/synth_2018.hdf5 --frac 0.1 --out ckpt/scratch10.pt --name ft_scratch_10
  python train.py fewshot  --data data/synth_2018.hdf5 --init ckpt/ft.pt --shots 5
  python train.py finetune --data data/synth_2018.hdf5 --arch vit --out ckpt/vit.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from datetime import UTC, datetime

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.dataset import IQDataset
from data.radioml import MODS_2018, load_2016, load_2018
from models.resnet1d import ResNet1D

HELD_OUT = ["32APSK", "128QAM", "OQPSK"]  # few-shot novel classes
BASE = [m for m in MODS_2018 if m not in HELD_OUT]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(arch: str, n_classes: int) -> nn.Module:
    if arch == "vit":
        from models.vit_spec import ViTSpec

        return ViTSpec(n_classes)
    return ResNet1D(n_classes)


def load_backbone(model: nn.Module, path: str) -> None:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck) if isinstance(ck, dict) and "model" in ck else ck
    sd = {k: v for k, v in sd.items() if not k.startswith("head")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded backbone from {path} (missing={len(missing)} unexpected={len(unexpected)})")


def run_epoch(model, dl, opt=None):
    model.train(opt is not None)
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(opt is not None):
            out = model(x)
            loss = nn.functional.cross_entropy(out, y)
            if opt:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        tot += len(y)
        correct += (out.argmax(1) == y).sum().item()
        loss_sum += loss.item() * len(y)
    return loss_sum / tot, correct / tot


@torch.no_grad()
def acc_vs_snr(model, X, y, snr, bs=512):
    model.eval()
    res = {}
    for s in sorted(np.unique(snr)):
        m = snr == s
        c = sum((model(x.to(dev)).argmax(1).cpu() == yy).sum().item() for x, yy in DataLoader(IQDataset(X[m], y[m]), bs))
        res[int(s)] = round(c / m.sum(), 4)
    return res


@torch.no_grad()
def confusion(model, X, y, n_classes, bs=512):
    model.eval()
    cm = np.zeros((n_classes, n_classes), int)
    for x, yy in DataLoader(IQDataset(X, y), bs):
        pred = model(x.to(dev)).argmax(1).cpu().numpy()
        np.add.at(cm, (yy.numpy(), pred), 1)
    return cm.tolist()


def fit(model, Xtr, ytr, Xva, yva, epochs, lr, bs, log, workers=4):
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tr = DataLoader(IQDataset(Xtr, ytr, augment=True), bs, shuffle=True, num_workers=workers, pin_memory=dev == "cuda", drop_last=len(ytr) > bs)
    va = DataLoader(IQDataset(Xva, yva), bs, num_workers=workers)
    best, state, history = -1.0, {k: v.clone() for k, v in model.state_dict().items()}, []
    for e in range(epochs):
        t0 = time.time()
        tl, ta = run_epoch(model, tr, opt)
        vl, vacc = run_epoch(model, va)
        sched.step()
        rec = dict(epoch=e, train_loss=round(tl, 4), train_acc=round(ta, 4), val_loss=round(vl, 4), val_acc=round(vacc, 4), seconds=round(time.time() - t0, 1))
        history.append(rec)
        log(e, rec)
        if vacc > best:
            best, state = vacc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return best, history


def export_onnx(model, L, path, classes):
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 2, L), path, input_names=["iq"], output_names=["logits"],
                      dynamic_axes={"iq": {0: "batch", 2: "length"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    json.dump(list(classes), open(path + ".classes.json", "w"))
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 2, L)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"iq": x.numpy()})[0]
    return float(np.abs(out - ref).max())


def provenance(a) -> dict:
    synthetic = None
    if a.data.endswith((".hdf5", ".h5")):
        with h5py.File(a.data, "r") as f:
            synthetic = bool(f.attrs.get("synthetic", False))
    elif "synth" in os.path.basename(a.data):
        synthetic = True
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = None
    return dict(data=a.data, synthetic=synthetic, device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                seed=a.seed, git=commit, timestamp=datetime.now(UTC).isoformat(timespec="seconds"), args=vars(a))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["pretrain", "finetune", "fewshot"])
    p.add_argument("--data", required=True)
    p.add_argument("--init")
    p.add_argument("--out", default="ckpt/model.pt")
    p.add_argument("--name", help="results/<name>.json (default: mode[_tag])")
    p.add_argument("--arch", choices=["resnet", "vit"], default="resnet")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--frac", type=float, default=1.0)
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--shots", type=int, default=5)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max_per_class", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mlflow", action="store_true")
    a = p.parse_args()
    seed_all(a.seed)
    t_start = time.time()
    name = a.name or a.mode
    if a.mlflow:
        import mlflow

        mlflow.start_run(run_name=name)
        mlflow.log_params(vars(a))

        def log(e, d):
            mlflow.log_metrics({k: v for k, v in d.items() if isinstance(v, (int, float))}, step=e)
            print(e, d, flush=True)
    else:

        def log(e, d):
            print(e, d, flush=True)

    if a.mode == "pretrain":
        X, y, snr, classes = load_2016(a.data, snr_min=-10)
    elif a.mode == "finetune":
        X, y, snr, classes = load_2018(a.data, classes=BASE, snr_min=-10, max_per_class=a.max_per_class, seed=a.seed)
    else:
        X, y, snr, classes = load_2018(a.data, classes=HELD_OUT, snr_min=0, max_per_class=500, seed=a.seed)
    Xtr, Xva, ytr, yva, _, sva = train_test_split(X, y, snr, test_size=0.2, stratify=y, random_state=a.seed)
    if a.frac < 1:
        Xtr, _, ytr, _ = train_test_split(Xtr, ytr, train_size=a.frac, stratify=ytr, random_state=a.seed)
    print(dict(mode=a.mode, n_train=len(ytr), n_val=len(yva), classes=len(classes), length=X.shape[2], device=dev), flush=True)

    model = build_model(a.arch, len(classes)).to(dev)
    if a.init:
        load_backbone(model, a.init)
    result = dict(mode=a.mode, name=name, arch=a.arch, frac=a.frac, init=a.init, n_train=int(len(ytr)), n_val=int(len(yva)),
                  classes=list(classes), length=int(X.shape[2]), **provenance(a))

    if a.mode == "fewshot":  # prototypical evaluation on frozen embeddings
        model.eval()
        rng = np.random.default_rng(a.seed)
        with torch.no_grad():
            E = torch.cat([model.embed(x.to(dev)) for x, _ in DataLoader(IQDataset(Xva, yva), 512)])
        accs = []
        for _ in range(a.episodes):
            protos = []
            for c in range(len(classes)):
                s = rng.choice(np.where(ytr == c)[0], a.shots, replace=False)
                with torch.no_grad():
                    protos.append(model.embed(torch.from_numpy(Xtr[s]).to(dev)).mean(0))
            pred = torch.cdist(E, torch.stack(protos)).argmin(1).cpu().numpy()
            accs.append(float((pred == yva).mean()))
        result.update(shots=a.shots, episodes=a.episodes, acc_mean=round(float(np.mean(accs)), 4), acc_std=round(float(np.std(accs)), 4),
                      chance=round(1 / len(classes), 4), seconds=round(time.time() - t_start, 1))
    else:
        if a.freeze:
            model.freeze_backbone(True)
        best, history = fit(model, Xtr, ytr, Xva, yva, a.epochs, a.lr, a.bs, log, a.workers)
        result.update(best_val_acc=round(best, 4), epochs=a.epochs, history=history, acc_vs_snr=acc_vs_snr(model, Xva, yva, sva),
                      confusion=confusion(model, Xva, yva, len(classes)), seconds=round(time.time() - t_start, 1))
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        torch.save(dict(model=model.state_dict(), arch=a.arch, classes=list(classes), length=int(X.shape[2])), a.out)
        if a.arch == "resnet":
            onnx_path = a.out.replace(".pt", ".onnx")
            result["onnx"] = dict(path=onnx_path, max_abs_diff_vs_torch=export_onnx(model, X.shape[2], onnx_path, classes))
        if a.mlflow:
            mlflow.log_metric("best_val_acc", best)
            mlflow.end_run()
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(f"results/{name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in result.items() if k not in ("history", "confusion", "args")}), flush=True)


if __name__ == "__main__":
    main()

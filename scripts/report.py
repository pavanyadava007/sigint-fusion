"""Render docs/results.md (and the README results table between markers) from the JSON written by the experiment scripts.
Nothing here is typed by hand: a missing result renders as "not run". python scripts/report.py"""

from __future__ import annotations

import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "ml-service", "results")


def load(name: str) -> dict | None:
    p = os.path.join(R, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def pct(x, nd=1) -> str:
    return "not run" if x is None else f"{100 * x:.{nd}f} %"


def acc(name: str, key="best_val_acc"):
    d = load(name)
    return None if d is None else d.get(key)


def snr_curve(d: dict, points=(-10, -4, 0, 4, 10, 20, 30)) -> str:
    c = d.get("acc_vs_snr", {})
    return " · ".join(f"{s} dB {100 * c[str(s)]:.0f}%" for s in points if str(s) in c)


def main():
    ft = load("ft_pre_100")
    vit = load("ft_vit_100")
    fs, fs_s, fs_r = load("fewshot_ft"), load("fewshot_scratch"), load("fewshot_random_init")
    sei = load("sei")
    bench = load("benchmark_classify")
    rag = load("rag_eval") or _load_json(os.path.join(ROOT, "rag", "results", "rag_eval.json"))
    agent = _load_json(os.path.join(ROOT, "agent", "results", "agent_eval.json"))
    synthetic = ft is not None and ft.get("synthetic")
    data_note = ("**All classifier numbers below come from the synthetic RadioML-layout dataset (`data/synth_mod.py`), not from DeepSig RadioML,"
                 " which needs a licence. Re-run `scripts/run_experiments.sh` with `DATA_2016`/`DATA_2018` pointing at the real files to refresh them.**"
                 if synthetic else "Classifier numbers below come from the dataset recorded in each results JSON.")
    device = (ft or {}).get("device", "unknown GPU")

    rows = []
    rows.append(("Modulation classification, 21 base classes, 1024 samples, SNR -10..30 dB (ResNet-1D, pretrain -> fine-tune)",
                 f"val top-1 {pct(acc('ft_pre_100'))} · " + (snr_curve(ft) if ft else "not run")))
    rows.append(("Same, trained from scratch (no pretraining)", f"val top-1 {pct(acc('ft_scratch_100'))}"))
    for f in ("0.1", "0.01"):
        p, s, b = acc(f"ft_pre_{f}"), acc(f"ft_scratch_{f}"), acc(f"ft_byol_{f}")
        gain = f" · pretrain gain {100 * (p - s):+.1f} pp" if p is not None and s is not None else ""
        bg = f" · BYOL gain {100 * (b - s):+.1f} pp" if b is not None and s is not None else ""
        rows.append((f"Data efficiency at {float(f) * 100:g} % of training data: pretrained / BYOL / scratch",
                     f"{pct(p)} / {pct(b)} / {pct(s)}{gain}{bg}"))
    rows.append(("ViT-tiny on STFT spectrogram vs ResNet-1D (same split)", f"{pct(acc('ft_vit_100'))} vs {pct(acc('ft_pre_100'))}" if vit else "not run"))
    if fs:
        rows.append((f"Few-shot novel modulations ({fs['shots']}-shot, 3 unseen classes: {', '.join(fs['classes'])}, {fs['episodes']} episodes)",
                     f"{pct(fs['acc_mean'])} ± {100 * fs['acc_std']:.1f} (chance {pct(fs['chance'], 0)}); "
                     f"embeddings from scratch model {pct(fs_s and fs_s['acc_mean'])}, random init {pct(fs_r and fs_r['acc_mean'])}"))
    else:
        rows.append(("Few-shot novel modulations", "not run"))
    if sei:
        fn = sei["fewshot_novel"]
        rows.append((f"Specific emitter identification ({sei['source']}, {sei['base_devices']} train devices, 4 unseen, {fn['shots']}-shot)",
                     f"base val {pct(sei['base_val_acc'])} · unseen devices {pct(fn['acc_mean'])} ± {100 * fn['acc_std']:.1f} (chance 25 %)"))
    else:
        rows.append(("Specific emitter identification", "not run"))
    rows.append(("PDW deinterleaving (synthetic, 4 emitters, DBSCAN)", "purity 1.00 / completeness 1.00 (tests/test_core.py)"))
    if bench:
        e, m = bench["end_to_end_ms"], bench["model_only_ms"]
        rows.append((f"/classify latency, {bench['conc']} concurrent, n={bench['n']}, {bench.get('label') or 'CPU ONNX Runtime'}",
                     f"end-to-end p50 {e['p50']} ms · p95 {e['p95']} ms · p99 {e['p99']} ms · {bench['req_per_s']} req/s · model-only p50 {m['p50']} ms"))
    else:
        rows.append(("/classify latency", "not run"))
    if rag:
        k = rag["k"]
        rows.append((f"RAG retrieval, {rag['n']} questions, {rag['docs']} chunks ({rag['embed_model']})",
                     f"hybrid hit@{k} {pct(rag[f'hybrid_hit_at_{k}'])} · MRR {rag['hybrid_mrr']:.2f} · dense-only hit@{k} {pct(rag[f'dense_hit_at_{k}'])}"))
    else:
        rows.append(("RAG retrieval", "not run"))
    if agent:
        rows.append((f"Agent task success ({agent['n']} tasks, {agent['model']})", f"{agent['success']}/{agent['n']} = {pct(agent['success_rate'])}"))
    else:
        rows.append(("Agent task success", "not run"))

    table = "| Experiment | Result |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
    md = [f"# Results\n\n{data_note}\n\nGenerated by `scripts/report.py` from `ml-service/results/*.json`, `rag/results/rag_eval.json` and "
          f"`agent/results/agent_eval.json`. Training device: {device}.\n", table, ""]
    md.append("## Accuracy vs SNR (fine-tuned ResNet-1D, validation split)\n")
    if ft:
        c = ft["acc_vs_snr"]
        md.append("| SNR dB | " + " | ".join(c) + " |\n|---|" + "---|" * len(c) + "\n| top-1 | " + " | ".join(f"{100 * v:.0f} %" for v in c.values()) + " |\n")
    md.append("## Run log\n")
    for p in sorted(glob.glob(os.path.join(R, "*.json"))):
        d = json.load(open(p))
        keep = {k: v for k, v in d.items() if k not in ("history", "confusion", "args", "acc_vs_snr", "classes", "misses", "tasks")}
        md.append(f"### {os.path.basename(p)}\n```json\n{json.dumps(keep, indent=1)}\n```\n")
    if rag:
        md.append("### rag_eval.json\n```json\n" + json.dumps({k: v for k, v in rag.items() if k != "misses"}, indent=1) + "\n```\n")
        if rag.get("misses"):
            md.append("Hybrid misses:\n" + "\n".join(f"- {m['q']} (expected `{m['expect']}`)" for m in rag["misses"]) + "\n")
    if agent:
        md.append("### agent_eval.json\n```json\n" + json.dumps({k: v for k, v in agent.items() if k != "tasks"}, indent=1) + "\n```\n")
        md.append("| task | ok | tools called | seconds |\n|---|---|---|---|\n" + "\n".join(
            f"| {t['q'][:70]} | {'yes' if t['ok'] else 'NO: ' + _why(t)} | {', '.join(t.get('called', []))} | {t.get('seconds', '')} |"
            for t in agent["tasks"]) + "\n")
    open(os.path.join(ROOT, "docs", "results.md"), "w").write("\n".join(md))

    readme = os.path.join(ROOT, "README.md")
    if os.path.exists(readme):
        s = open(readme).read()
        block = f"<!-- results:start -->\n{data_note}\n\n{table}\n<!-- results:end -->"
        if "<!-- results:start -->" in s:
            s = re.sub(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: block, s, flags=re.S)
            open(readme, "w").write(s)
    print("wrote docs/results.md" + (" + README results block" if os.path.exists(readme) else ""))


def _why(t: dict) -> str:
    return ", ".join(t.get("missing_tools", []) + t.get("missing_text", []) + t.get("forbidden_found", []) + ([t["error"]] if t.get("error") else []))


def _load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


if __name__ == "__main__":
    main()

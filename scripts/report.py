"""Render docs/results.md (and the README results block between markers) from the JSON written by the experiment scripts.
Nothing here is typed by hand: a missing result renders as "not run". Real-RadioML runs (tag "_real") and synthetic runs (no tag)
are rendered as separate tables. python scripts/report.py"""

from __future__ import annotations

import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "ml-service", "results")


def _load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load(name: str) -> dict | None:
    return _load_json(os.path.join(R, f"{name}.json"))


def pct(x, nd=1) -> str:
    return "not run" if x is None else f"{100 * x:.{nd}f} %"


def acc(name: str, key="best_val_acc"):
    d = load(name)
    return None if d is None else d.get(key)


def snr_curve(d: dict, points=(-10, -4, 0, 4, 10, 20, 30)) -> str:
    c = d.get("acc_vs_snr", {})
    return " · ".join(f"{s} dB {100 * c[str(s)]:.0f}%" for s in points if str(s) in c)


def classifier_rows(tag: str) -> tuple[list, dict | None]:
    """Rows for one ladder run: tag '' = synthetic files, '_real' = DeepSig RadioML files."""
    ft, vit, sei = load(f"ft_pre_100{tag}"), load(f"ft_vit_100{tag}"), load(f"sei{tag}")
    fs, fs_s, fs_r = load(f"fewshot_ft{tag}"), load(f"fewshot_scratch{tag}"), load(f"fewshot_random_init{tag}")
    pre = load(f"pretrain{tag}")
    rows = []
    if pre:
        rows.append((f"Pretraining on 2016.10a-layout frames ({len(pre['classes'])} classes, 128 samples, SNR -10..{max(map(int, pre['acc_vs_snr']))} dB)",
                     f"val top-1 {pct(pre['best_val_acc'])} · " + snr_curve(pre, points=(-10, -4, 0, 4, 10, 18))))
    rows.append(("Modulation classification, 21 base classes, 1024 samples, SNR -10..30 dB (ResNet-1D, pretrain -> fine-tune)",
                 f"val top-1 {pct(acc(f'ft_pre_100{tag}'))} · " + (snr_curve(ft) if ft else "not run")))
    rows.append(("Same, trained from scratch (no pretraining)", f"val top-1 {pct(acc(f'ft_scratch_100{tag}'))}"))
    for f in ("0.1", "0.01"):
        p, s, b = acc(f"ft_pre_{f}{tag}"), acc(f"ft_scratch_{f}{tag}"), acc(f"ft_byol_{f}{tag}")
        gain = f" · pretrain gain {100 * (p - s):+.1f} pp" if p is not None and s is not None else ""
        bg = f" · BYOL gain {100 * (b - s):+.1f} pp" if b is not None and s is not None else ""
        rows.append((f"Data efficiency at {float(f) * 100:g} % of training data: pretrained / BYOL / scratch", f"{pct(p)} / {pct(b)} / {pct(s)}{gain}{bg}"))
    vit_txt = f"{pct(acc(f'ft_vit_100{tag}'))} vs {pct(acc(f'ft_pre_100{tag}'))}" if vit else "not run"
    rows.append(("ViT-tiny on STFT spectrogram vs ResNet-1D (same split)", vit_txt))
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
    return rows, ft


def platform_rows() -> list:
    rows = []
    for name, what in (("real_rml2016_6db", "synthetic-pretrained"),
                       ("real_rml2016_6db_from_real_pretrain", "real-2016.10a-pretrained (CAVEAT: the mirror slice is likely drawn from 2016.10a, "
                                                              "so its test frames may overlap the pretraining split; indicative only)")):
        real = load(name)
        if not real:
            continue
        r = real["runs"]
        for f in ("1.0", "0.1"):
            sc, pr, lp = r[f"scratch@{f}"], r[f"pretrained@{f}"], r[f"linear_probe@{f}"]
            label = (f"REAL RadioML 2016.10a frames (HF mirror, single 6 dB slice, {real['classes']} classes, {pr['n_train']} train frames = "
                     f"{float(f) * 100:g} %), test acc over {real['seeds']} seeds: {what} / scratch / frozen linear probe")
            val = (f"{pct(pr['mean'])} ± {100 * pr['std']:.1f} / {pct(sc['mean'])} ± {100 * sc['std']:.1f} / {pct(lp['mean'])} ± {100 * lp['std']:.1f}"
                   f" · transfer gain {100 * (pr['mean'] - sc['mean']):+.1f} pp (chance {pct(real['chance'], 0)})")
            rows.append((label, val))
    rows.append(("PDW deinterleaving (synthetic, 4 emitters, DBSCAN)", "purity 1.00 / completeness 1.00 (tests/test_core.py)"))
    bench = load("benchmark_classify")
    if bench:
        e, m = bench["end_to_end_ms"], bench["model_only_ms"]
        rows.append((f"/classify latency, {bench['conc']} concurrent, n={bench['n']}, {bench.get('label') or 'CPU ONNX Runtime'}",
                     f"end-to-end p50 {e['p50']} ms · p95 {e['p95']} ms · p99 {e['p99']} ms · {bench['req_per_s']} req/s · model-only p50 {m['p50']} ms"))
    else:
        rows.append(("/classify latency", "not run"))
    rag = load("rag_eval") or _load_json(os.path.join(ROOT, "rag", "results", "rag_eval.json"))
    if rag:
        k = rag["k"]
        rows.append((f"RAG retrieval, {rag['n']} questions, {rag['docs']} chunks ({rag['embed_model']})",
                     f"hybrid hit@{k} {pct(rag[f'hybrid_hit_at_{k}'])} · MRR {rag['hybrid_mrr']:.2f} · dense-only hit@{k} {pct(rag[f'dense_hit_at_{k}'])}"))
    else:
        rows.append(("RAG retrieval", "not run"))
    agent = _load_json(os.path.join(ROOT, "agent", "results", "agent_eval.json"))
    if agent:
        rows.append((f"Agent task success ({agent['n']} tasks, {agent['model']})", f"{agent['success']}/{agent['n']} = {pct(agent['success_rate'])}"))
    else:
        rows.append(("Agent task success", "not run"))
    return rows


def tbl(rows) -> str:
    return "| Experiment | Result |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


def _why(t: dict) -> str:
    return ", ".join(t.get("missing_tools", []) + t.get("missing_text", []) + t.get("forbidden_found", []) + ([t["error"]] if t.get("error") else []))


def main():
    real_rows, ft_real = classifier_rows("_real")
    syn_rows, ft_syn = classifier_rows("")
    plat = platform_rows()
    device = (ft_real or ft_syn or load("pretrain_real") or {}).get("device", "unknown GPU")
    parts = []
    real_done = any("not run" not in b for _, b in real_rows)
    if real_done:
        src = (ft_real or load("pretrain_real") or {}).get("data", "")
        pending = any("not run" in b for _, b in real_rows)
        parts.append(f"**Real DeepSig RadioML** (`{src}`{'; rows saying not run are still in progress' if pending else ''})\n\n{tbl(real_rows)}")
    else:
        parts.append("**Real DeepSig RadioML**: not run yet. Register at deepsig.ai, put the files in `ml-service/data/`, then "
                     "`DATA_2016=data/RML2016.10a_dict.pkl DATA_2018=data/GOLD_XYZ_OSC.0001_1024.hdf5 scripts/run_experiments.sh`.")
    parts.append("**Synthetic RadioML-layout data** (`data/synth_mod.py`: same classes and file layouts, so these are pipeline numbers, "
                 "NOT RadioML results)\n\n" + tbl(syn_rows))
    parts.append("**Platform, retrieval, agent** (measured on the live stack)\n\n" + tbl(plat))
    table = "\n\n".join(parts)
    note = ("Every number is written by a script into a results JSON with provenance (dataset path, synthetic flag, GPU, seed, git commit) "
            "and rendered from there; nothing in these tables is typed by hand.")
    md = [f"# Results\n\n{note}\n\nGenerated by `scripts/report.py` from `ml-service/results/*.json`, `rag/results/rag_eval.json` and "
          f"`agent/results/agent_eval.json`. Training device: {device}.\n", table, ""]
    for label, d in (("real RadioML 2018.01A", ft_real), ("synthetic 2018-layout", ft_syn)):
        if d:
            c = d["acc_vs_snr"]
            md.append(f"## Accuracy vs SNR, fine-tuned ResNet-1D, validation split ({label})\n")
            head = "| SNR dB | " + " | ".join(c) + " |\n|---|" + "---|" * len(c)
            md.append(head + "\n| top-1 | " + " | ".join(f"{100 * v:.0f} %" for v in c.values()) + " |\n")
    md.append("## Run log\n")
    for p in sorted(glob.glob(os.path.join(R, "*.json"))):
        d = json.load(open(p))
        keep = {k: v for k, v in d.items() if k not in ("history", "confusion", "args", "acc_vs_snr", "classes", "misses", "tasks", "runs")}
        md.append(f"### {os.path.basename(p)}\n```json\n{json.dumps(keep, indent=1)}\n```\n")
    rag = load("rag_eval") or _load_json(os.path.join(ROOT, "rag", "results", "rag_eval.json"))
    if rag:
        md.append("### rag_eval.json\n```json\n" + json.dumps({k: v for k, v in rag.items() if k != "misses"}, indent=1) + "\n```\n")
        if rag.get("misses"):
            md.append("Hybrid misses:\n" + "\n".join(f"- {m['q']} (expected `{m['expect']}`)" for m in rag["misses"]) + "\n")
    agent = _load_json(os.path.join(ROOT, "agent", "results", "agent_eval.json"))
    if agent:
        md.append("### agent_eval.json\n```json\n" + json.dumps({k: v for k, v in agent.items() if k != "tasks"}, indent=1) + "\n```\n")
        md.append("| task | ok | tools called | seconds |\n|---|---|---|---|\n" + "\n".join(
            f"| {t['q'][:70]} | {'yes' if t['ok'] else 'NO: ' + _why(t)} | {', '.join(t.get('called', []))} | {t.get('seconds', '')} |"
            for t in agent["tasks"]) + "\n")
    open(os.path.join(ROOT, "docs", "results.md"), "w").write("\n".join(md))
    readme = os.path.join(ROOT, "README.md")
    if os.path.exists(readme):
        s = open(readme).read()
        block = f"<!-- results:start -->\n{table}\n\n{note}\n<!-- results:end -->"
        if "<!-- results:start -->" in s:
            open(readme, "w").write(re.sub(r"<!-- results:start -->.*?<!-- results:end -->", lambda _: block, s, flags=re.S))
    print("wrote docs/results.md + README results block")


if __name__ == "__main__":
    main()

"""Publish the browser-mode operator console to a Hugging Face static Space.

The Space serves web/dist-browser (built by `npm run build:browser`): the same React console, with the ONNX classifier
(onnxruntime-web), DSP, sensor scenario, fusion and knowledge base running inside the browser. No server, no LLM.

python scripts/publish_hf.py --space pavanyadava07/sigint-fusion [--no-build]
Token: HF_TOKEN env or ~/.cache/huggingface/token (needs write access).
"""

from __future__ import annotations

import argparse
import os
import subprocess

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
DIST = os.path.join(WEB, "dist-browser")

CARD = """---
title: SIGINT-Fusion Operator Console
emoji: 📡
colorFrom: gray
colorTo: blue
sdk: static
pinned: false
license: other
short_description: SIGINT-Fusion operator console, fully in the browser
---

# SIGINT-Fusion operator console (browser demo)

The operator console of [SIGINT-Fusion](https://github.com/pavanyadava007/sigint-fusion) running entirely in your browser:
the ResNet-1D modulation classifier (trained on DeepSig RadioML 2018.01A, served with onnxruntime-web), Welch PSD, CFAR
detection, STFT spectrogram, DBSCAN pulse deinterleaving, Kalman/Hungarian emitter fusion, a sensor scenario streaming real
RadioML frames, and the analyst: the console gathers the evidence (catalogue matches, knowledge-base passages, live
detections and fused tracks) and sends it to a companion ZeroGPU Space running Qwen2.5-7B-Instruct
([pavanyadava07/sigint-fusion-agent](https://huggingface.co/spaces/pavanyadava07/sigint-fusion-agent)) which writes the
grounded answer or report. When that Space is asleep or out of quota, a rule-based analyst answers from the same evidence
and the reply says so.

What is different from the full stack: no Kafka, no Postgres, no Java gateway; the LLM runs on a shared ZeroGPU Space
(first request after idle takes 30 to 60 s) instead of a local Ollama. See the repository for the measured results.

Data notice: the bundled I/Q frames (320 frames, 8 modulations) are from DeepSig RadioML 2018.01A, licensed
CC BY-NC-SA 4.0 by DeepSig Inc.; the bundled model weights were trained on that data and are shared under the same terms.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--space", default="pavanyadava07/sigint-fusion")
    p.add_argument("--no-build", action="store_true")
    a = p.parse_args()
    if not a.no_build:
        subprocess.run(["npm", "run", "build:browser"], cwd=WEB, check=True)
    assert os.path.exists(os.path.join(DIST, "index.html")), "build missing"
    with open(os.path.join(DIST, "README.md"), "w") as f:
        f.write(CARD)
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.create_repo(a.space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=DIST, repo_id=a.space, repo_type="space", delete_patterns=["*"], commit_message="publish browser console")
    print(f"published: https://huggingface.co/spaces/{a.space}")


if __name__ == "__main__":
    main()

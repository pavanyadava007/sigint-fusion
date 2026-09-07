---
title: SIGINT-Fusion Analyst Backend
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Qwen2.5-7B analyst backend for the SIGINT-Fusion console
---

# SIGINT-Fusion analyst backend

Qwen2.5-7B-Instruct on ZeroGPU, serving `POST /ask {question, context, report}` for the browser demo at
[pavanyadava07/sigint-fusion](https://huggingface.co/spaces/pavanyadava07/sigint-fusion). The console gathers the tool output
(catalogue matches, knowledge-base passages, live detections, fused tracks) and this Space writes the grounded answer or report.
Source: [github.com/pavanyadava007/sigint-fusion](https://github.com/pavanyadava007/sigint-fusion) (`deploy/hf_space_agent`).

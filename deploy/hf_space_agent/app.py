"""SIGINT-Fusion analyst backend for the browser demo: Qwen2.5-7B-Instruct on a Hugging Face ZeroGPU Space.

The browser console does the retrieval and tool work itself (catalogue lookup, knowledge-base search, live detections and
EOB) and sends the question plus that context here; the model writes the grounded answer or report. Exposed as
POST /ask {question, context, report:bool} -> {answer, model, seconds, gpu:bool} with CORS open, plus a Gradio UI for manual use.
Runs without the `spaces` package too (plain GPU or CPU), which is how it is tested locally.
"""

from __future__ import annotations

import json
import os
import time

import gradio as gr
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces

    GPU = spaces.GPU
    ZERO = True
except Exception:  # noqa: BLE001
    ZERO = False

    def GPU(f=None, **_):  # noqa: N802
        return f if f else (lambda g: g)


MODEL_ID = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "700"))
SYSTEM = """You are a SIGINT analyst assistant inside an information fusion system. You are given tool output gathered by the
operator console (recent detections, fused emitter tracks, catalogue matches, knowledge-base passages with [source] tags).
Rules: answer only from that context; cite knowledge-base passages as [source]; state classifier confidence and mark
identifications below 0.6 as tentative; never invent emitters, frequencies, detections or confidence numbers."""
PLAIN = ("Answer in plain markdown prose of at most six sentences and NO headings or tables: give the identification, the evidence "
         "(catalogue match, track parameters, cited [source] passages) and any caveat.")
REPORT = ("Write an intelligence report with exactly these markdown sections: ## Summary, ## Emitters (a markdown table), "
          "## Assessment, ## Confidence, ## Recommended actions.")

_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto" if torch.cuda.is_available() else None)
_model.eval()


@GPU(duration=90)
def generate(question: str, context: str, report: bool) -> str:
    user = f"Tool output:\n{context[:12000]}\n\nOperator question: {question}\n\n{REPORT if report else PLAIN}"
    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=user)]
    enc = _tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(_model.device)
    with torch.no_grad():
        out = _model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None)
    return _tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class Ask(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    context: str = Field("", max_length=20000)
    report: bool = False


api = FastAPI(title="SIGINT-Fusion analyst backend")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@api.post("/ask")
def ask(a: Ask):
    t0 = time.time()
    return dict(answer=generate(a.question, a.context, a.report), model=MODEL_ID, seconds=round(time.time() - t0, 2), gpu=torch.cuda.is_available(), zerogpu=ZERO)


@api.get("/health")
def health():
    return dict(status="ok", model=MODEL_ID, gpu=torch.cuda.is_available(), zerogpu=ZERO)


def ui_fn(question, context):
    return generate(question, context, "report" in question.lower())


demo = gr.Interface(ui_fn, [gr.Textbox(label="Operator question", value="What is transmitting around 9400 MHz? Identify it."),
                            gr.Textbox(label="Tool output (JSON/text the console would send)", lines=8, value=json.dumps(dict(
                                catalogue=[dict(name="Generic X-band marine navigation radar", rf_min_mhz=9300, rf_max_mhz=9500, modulation="pulse", pri_us=500, pw_us=0.5)],
                                passages=["[frequency_allocations.md] 9300-9500 MHz: maritime radionavigation, X-band marine navigation radar ..."])))],
                    gr.Markdown(label="Answer"), title="SIGINT-Fusion analyst backend",
                    description="Backend for the browser console at huggingface.co/spaces/pavanyadava07/sigint-fusion. POST /ask for JSON.",
                    flagging_mode="never")
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))

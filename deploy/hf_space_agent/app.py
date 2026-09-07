"""SIGINT-Fusion analyst backend for the browser demo: Qwen2.5-7B-Instruct on a Hugging Face ZeroGPU Space.

The browser console does the retrieval and tool work itself (catalogue lookup, knowledge-base search, live detections and
fused tracks) and sends the question plus that evidence here; the model writes the grounded answer or report.
API (Gradio): fn "ask" with inputs (question, context, report) -> JSON string {answer, model, seconds, gpu, zerogpu};
call it with @gradio/client / gradio_client, e.g. client.predict(question, context, report, api_name="/ask").
Runs without the `spaces` package too (plain GPU or CPU), which is how it is tested locally.
"""

try:  # ZeroGPU: must be imported before torch/CUDA
    import spaces

    ZERO = True
except Exception:  # noqa: BLE001
    spaces = None
    ZERO = False

import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import gradio as gr  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW = int(os.getenv("MAX_NEW_TOKENS", "700"))
SYSTEM = """You are a SIGINT analyst assistant inside an information fusion system. You are given tool output gathered by the
operator console (recent detections, fused emitter tracks, catalogue matches, knowledge-base passages with [source] tags).
Rules: answer only from that context; cite knowledge-base passages as [source]; state classifier confidence and mark
identifications below 0.6 as tentative; never invent emitters, frequencies, detections or confidence numbers."""
PLAIN = ("Answer in plain markdown prose of at most six sentences and NO headings or tables: give the identification, the evidence "
         "(catalogue match, track parameters, cited [source] passages) and any caveat. Quote a confidence value only if one appears "
         "in the tool output; otherwise describe confidence in words (high when catalogue, track and passages agree).")
REPORT = ("Write an intelligence report with exactly these markdown sections: ## Summary, ## Emitters (a markdown table), "
          "## Assessment, ## Confidence, ## Recommended actions.")

_tok = AutoTokenizer.from_pretrained(MODEL_ID)
_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
if torch.cuda.is_available() or ZERO:
    _model = _model.to("cuda")  # on ZeroGPU the weights are staged to the shared GPU at startup
_model.eval()


def _generate(question: str, context: str, report: bool) -> str:
    user = f"Tool output:\n{context[:12000]}\n\nOperator question: {question}\n\n{REPORT if report else PLAIN}"
    msgs = [dict(role="system", content=SYSTEM), dict(role="user", content=user)]
    enc = _tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(_model.device)
    with torch.no_grad():
        out = _model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None)
    return _tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


if ZERO:
    _generate = spaces.GPU(duration=90)(_generate)


def ask(question: str, context: str = "", report: bool = False) -> str:
    t0 = time.time()
    question = (question or "").strip()[:4000]
    if not question:
        return json.dumps(dict(error="empty question"))
    answer = _generate(question, (context or "")[:20000], bool(report))
    return json.dumps(dict(answer=answer, model=MODEL_ID, seconds=round(time.time() - t0, 2), gpu=torch.cuda.is_available(), zerogpu=ZERO))


EXAMPLE_CTX = json.dumps(dict(
    catalogue=[dict(name="Generic X-band marine navigation radar", rf_min_mhz=9300, rf_max_mhz=9500, modulation="pulse", pri_us=500, pw_us=0.5)],
    tracks=[dict(track_id=7, rf_mhz=9410.0, aoa=118.1, sources=["resm"], hits=196, modulation="pulse", pri_type="stagger", pri_us=648)],
    passages=["[frequency_allocations.md] 9300-9500 MHz: maritime radionavigation, X-band marine navigation radar, near 9410 MHz."]), indent=1)

demo = gr.Interface(
    ask,
    [gr.Textbox(label="Operator question", value="What is transmitting around 9400 MHz? Identify it."),
     gr.Textbox(label="Tool output gathered by the console (JSON or text)", lines=10, value=EXAMPLE_CTX),
     gr.Checkbox(label="Report format", value=False)],
    gr.Textbox(label="JSON result", lines=14),
    title="SIGINT-Fusion analyst backend (Qwen2.5-7B-Instruct)",
    description="Backend for the browser console at huggingface.co/spaces/pavanyadava07/sigint-fusion. "
                "API: client.predict(question, context, report, api_name='/ask').",
    api_name="ask",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), ssr_mode=False)

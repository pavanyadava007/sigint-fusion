"""LangGraph ReAct SIGINT analyst. LLM: any OpenAI-compatible endpoint (Ollama / vLLM local, or OpenAI) via LLM_BASE_URL / LLM_MODEL / OPENAI_API_KEY."""

from __future__ import annotations

import json
import os
import re
import time

import httpx
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from tools.core import TOOLS

SYSTEM = """You are a SIGINT analyst assistant inside an information fusion system. Working method:
1. For questions about the current picture ("what is transmitting", "what do we see", "summarise detections", "order of battle"),
   call recent_detections AND current_tracks BEFORE answering.
2. Whenever a frequency is mentioned or observed (e.g. "9400 MHz", "2.8 GHz"), ALWAYS call query_emitter_db with rf_mhz in MHz
   (convert GHz to MHz) and also search_docs for the band allocation; cite sources in [brackets].
3. State classifier confidence explicitly; below 0.6 mark the identification as 'tentative'.
4. Only when the operator explicitly asks for a report (write / produce / generate a report): write it with the sections
   Summary / Emitters (markdown table) / Assessment / Confidence / Recommended actions, call save_report with the full markdown,
   and then give the report's key findings in your final answer as well (never reply with just "saved").
   Identification questions ("what is transmitting", "what is it") are answered directly with the emitter name and evidence; do NOT save a report.
5. For definitions, doctrine, procedures or band-allocation questions, call search_docs first and cite the [source]; do not answer
   from memory alone.
6. Answer concisely in markdown. Never invent emitter names, frequencies or detections that are not in tool output.
   If tools return nothing, say so."""

REPORT_REQUEST = re.compile(r"\b(write|produce|generate|create|make|draft|prepare|compile|save)\b[^.?!]*\breport\b|\breport on\b", re.I)

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")


def build():
    # trust_env=False: internal endpoints must never be routed through an HTTP proxy injected by the container runtime
    llm = ChatOpenAI(model=LLM_MODEL, base_url=LLM_BASE_URL, api_key=os.getenv("OPENAI_API_KEY", "ollama"), temperature=0, timeout=300, max_retries=2,
                     http_async_client=httpx.AsyncClient(trust_env=False, timeout=300), http_client=httpx.Client(trust_env=False, timeout=300))
    tools = [StructuredTool.from_function(coroutine=f, name=f.__name__, description=f.__doc__) for f in TOOLS]
    return create_react_agent(llm, tools, prompt=SYSTEM, checkpointer=MemorySaver())


def trace_of(messages, since: int) -> list[dict]:
    """Tool calls and results from this turn, for the operator console."""
    calls = {}
    for m in messages[since:]:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                calls[tc["id"]] = dict(tool=tc["name"], args=tc["args"], result_preview=None)
        elif isinstance(m, ToolMessage) and m.tool_call_id in calls:
            calls[m.tool_call_id]["result_preview"] = str(m.content)[:400]
    return list(calls.values())


async def ask(graph, text: str, thread: str = "default", recursion_limit: int = 24) -> dict:
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": recursion_limit}
    before = len((await graph.aget_state(cfg)).values.get("messages", []))
    t0 = time.time()
    out = await graph.ainvoke({"messages": [("user", text)]}, config=cfg)
    msgs = out["messages"]
    if REPORT_REQUEST.search(text) and not any(t["tool"] == "save_report" for t in trace_of(msgs, before)):
        # verification step: a report task is only complete once the report is persisted
        nudge = ("You have not saved the report. Call save_report now with the complete markdown report "
                 "(Summary / Emitters / Assessment / Confidence / Recommended actions), then confirm.")
        out = await graph.ainvoke({"messages": [("user", nudge)]}, config=cfg)
        msgs = out["messages"]
    answer = msgs[-1].content if isinstance(msgs[-1].content, str) else json.dumps(msgs[-1].content)
    return dict(answer=answer, trace=trace_of(msgs, before), seconds=round(time.time() - t0, 2), model=LLM_MODEL)

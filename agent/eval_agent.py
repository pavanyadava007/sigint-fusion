"""Agent task-success eval. tasks.jsonl: {"q":..., "must_call":[tool names], "must_call_any":[...], "must_contain":[substrings],
"must_contain_any":[...], "must_not_contain":[...]}.
A task succeeds when every required tool was called and the final answer contains all required (and none of the forbidden) substrings.
Writes results/agent_eval.json with per-task detail.

python eval_agent.py tasks.jsonl
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime

from agent import LLM_MODEL, ask, build

HERE = os.path.dirname(os.path.abspath(__file__))


async def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tasks.jsonl")
    tasks = [json.loads(line) for line in open(path) if line.strip()]
    g = build()
    detail = []
    for i, t in enumerate(tasks):
        try:
            res = await ask(g, t["q"], thread=f"eval-{i}")
            called = [x["tool"] for x in res["trace"]]
            ans = res["answer"].lower()
            missing_tools = [x for x in t.get("must_call", []) if x not in called]
            missing_text = [s for s in t.get("must_contain", []) if s.lower() not in ans]
            forbidden = [s for s in t.get("must_not_contain", []) if s.lower() in ans]
            if t.get("must_call_any") and not any(x in called for x in t["must_call_any"]):
                missing_tools.append("any of " + "|".join(t["must_call_any"]))
            if t.get("must_contain_any") and not any(s.lower() in ans for s in t["must_contain_any"]):
                missing_text.append("any of " + "|".join(t["must_contain_any"]))
            ok = not (missing_tools or missing_text or forbidden)
            detail.append(dict(q=t["q"], ok=ok, called=called, missing_tools=missing_tools, missing_text=missing_text, forbidden_found=forbidden,
                               seconds=res["seconds"], answer=res["answer"][:600]))
        except Exception as e:  # noqa: BLE001
            detail.append(dict(q=t["q"], ok=False, error=f"{type(e).__name__}: {e}"))
        print(json.dumps({k: v for k, v in detail[-1].items() if k != "answer"}), flush=True)
    n_ok = sum(d["ok"] for d in detail)
    res = dict(n=len(tasks), success=n_ok, success_rate=round(n_ok / len(tasks), 3), model=LLM_MODEL,
               timestamp=datetime.now(UTC).isoformat(timespec="seconds"), tasks=detail)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "agent_eval.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "tasks"}))


if __name__ == "__main__":
    asyncio.run(main())

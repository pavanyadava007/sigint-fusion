# agent

LangGraph ReAct analyst over the fusion database, the knowledge base and the ML service.

- `tools/core.py`: `search_docs`, `query_emitter_db`, `recent_detections`, `current_tracks`, `classify_signal`, `deinterleave`, `save_report`, `list_reports`. Shared by the HTTP API and the MCP server.
- `agent.py`: `build()` creates the graph (any OpenAI-compatible LLM via `LLM_BASE_URL` / `LLM_MODEL` / `OPENAI_API_KEY`; default Ollama qwen2.5:7b), `ask()` returns the answer plus the tool-call trace of that turn. Conversation memory is per thread id.
- `api.py`: `POST /ask {question, thread}` -> `{answer, trace, seconds, model}`; `GET /reports`; `GET /health` probes the LLM. Warms up the LLM and the embedding model at start.
- `mcp_server.py`: the same tools over MCP stdio for desktop-assistant or IDE clients:
  `{"mcpServers": {"sigint-fusion": {"command": "python", "args": ["/path/agent/mcp_server.py"], "env": {"PG": "...", "ML_URL": "http://localhost:8000"}}}}`
- `eval_agent.py`: `tasks.jsonl` (10 tasks: required tool calls, required and forbidden answer substrings) -> `results/agent_eval.json`.

```bash
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b uvicorn api:app --port 8001
python eval_agent.py tasks.jsonl
```
The Docker image bakes the embedding model in at build time, so the container never needs Hugging Face at runtime.

# Agent API + MCP server. Build context is the repo root so rag/ (embeddings, retrieval) is included.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 EMBED_CACHE=/models
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* && useradd -m -u 10001 app
WORKDIR /app
COPY agent/requirements.txt .
RUN pip install -r requirements.txt
# bake the embedding model into the image so the container never needs Hugging Face at runtime
RUN mkdir -p /models && python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models')" && chown -R app:app /models
COPY --chown=app:app rag/embeddings.py rag/retrieval.py rag/schema.sql rag/ingest.py rag/eval.py rag/questions.jsonl ./rag/
COPY --chown=app:app rag/corpus ./rag/corpus
COPY --chown=app:app agent/*.py ./
COPY --chown=app:app agent/tools ./tools
COPY --chown=app:app agent/tasks.jsonl ./
USER app
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 CMD curl -fsS http://127.0.0.1:8001/health || exit 1
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]

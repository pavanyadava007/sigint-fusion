"""Local embedding model shared by ingest, eval and the agent (fastembed, ONNX, no API key).

BAAI/bge-small-en-v1.5: 384-d; queries get the recommended instruction prefix, documents do not.
Set EMBED_MODEL to switch models (the pgvector column is vector(384)); EMBED_CACHE points at the model cache directory.
"""

from __future__ import annotations

import os

MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model: str = MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model, cache_dir=os.getenv("EMBED_CACHE"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        return next(self.model.embed([QUERY_PREFIX + text])).tolist()

# rag

Knowledge base for the analyst agent: pgvector schema, corpus, ingest, hybrid retrieval and its eval.

- `schema.sql`: all platform tables (detections, spectra, tracks, emitters, docs with HNSW + GIN indexes, reports). Applied by the Postgres container on first start and idempotently by `ingest.py`.
- `corpus/`: `frequency_allocations.md` (public ITU/ICAO/IMO band facts), `sigint_glossary.md`, `analysis_procedures.md`, `emitters_open.json` (24 generic open-source catalogue entries). Add your own markdown or JSON files and re-run ingest.
- `embeddings.py`: BAAI/bge-small-en-v1.5 via fastembed (384-d, ONNX, local). Queries get the bge instruction prefix.
- `retrieval.py`: dense cosine + Postgres full-text + exact numeric literal + catalogue band containment (a frequency mentioned with a unit is matched against `emitters` bands), fused with reciprocal rank fusion.
- `ingest.py`: heading-aware word-window chunking (180 words, 40 overlap), catalogue entries become one chunk each and are upserted into `emitters`. Re-running replaces each source's chunks.
- `eval.py`: `questions.jsonl` (53 questions with an expected substring) -> hit@5 and MRR for dense-only and hybrid, written to `results/rag_eval.json`.

```bash
python ingest.py                          # PG=postgresql://sigint:sigint@localhost:5432/sigint
python eval.py questions.jsonl --k 5
python -m pytest -q tests
```

-- SIGINT-Fusion schema. Applied by the Postgres container on first start and (idempotently) by rag/ingest.py.
CREATE EXTENSION IF NOT EXISTS vector;

-- classifier / deinterleaver outputs
CREATE TABLE IF NOT EXISTS detections (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  sensor_id TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'comint',          -- comint (I/Q classifier) | resm (PDW deinterleaver)
  rf_mhz REAL,
  aoa REAL,
  modulation TEXT,
  confidence REAL,
  latency_ms REAL,
  pw_us REAL,
  pri_us REAL,
  pri_type TEXT
);
CREATE INDEX IF NOT EXISTS detections_ts ON detections (ts DESC);
CREATE INDEX IF NOT EXISTS detections_sensor_ts ON detections (sensor_id, ts DESC);

-- latest spectra per sensor for the waterfall (ring of the last N rows per sensor is pruned by the consumer)
CREATE TABLE IF NOT EXISTS spectra (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  sensor_id TEXT NOT NULL,
  rf_mhz REAL,
  levels REAL[] NOT NULL
);
CREATE INDEX IF NOT EXISTS spectra_sensor_ts ON spectra (sensor_id, ts DESC);

-- fused Electronic Order of Battle snapshot (one row per live track, upserted by the consumer)
CREATE TABLE IF NOT EXISTS tracks (
  track_id INTEGER PRIMARY KEY,
  updated TIMESTAMPTZ NOT NULL DEFAULT now(),
  first_seen TIMESTAMPTZ,
  last_seen TIMESTAMPTZ,
  rf_mhz REAL,
  aoa REAL,
  rf_rate REAL,
  aoa_rate REAL,
  sources TEXT[],
  sensors TEXT[],
  hits INTEGER,
  misses INTEGER,
  modulation TEXT,
  label_agreement REAL,
  pw_us REAL,
  pri_us REAL,
  pri_type TEXT
);

-- open emitter catalogue
CREATE TABLE IF NOT EXISTS emitters (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT,
  rf_min_mhz REAL,
  rf_max_mhz REAL,
  pri_us REAL,
  pw_us REAL,
  modulation TEXT,
  notes TEXT,
  UNIQUE (name)
);

-- RAG corpus: dense (pgvector, cosine, HNSW) + lexical (tsvector, GIN) for hybrid retrieval
CREATE TABLE IF NOT EXISTS docs (
  id SERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  chunk_no INTEGER NOT NULL DEFAULT 0,
  chunk TEXT NOT NULL,
  embedding vector(384),
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk)) STORED,
  UNIQUE (source, chunk_no)
);
CREATE INDEX IF NOT EXISTS docs_emb ON docs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS docs_tsv ON docs USING gin (tsv);

-- analyst reports written by the agent
CREATE TABLE IF NOT EXISTS reports (
  id SERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  question TEXT,
  report TEXT,
  thread TEXT
);

# Architecture

```
                 HTTP (JSON)            Kafka (keyed by sensor)               Postgres / pgvector
 edge sensors ──────────────▶ gateway-java ──────────────▶ ml-consumer ───────────▶ detections, spectra, tracks
 (I/Q chunks,                 Spring Boot 3                ONNX classifier                 ▲          ▲
  PDW batches)                validation, metrics          DBSCAN deinterleaver            │          │
                                                           Kalman + Hungarian fusion       │          │
                                                                                            │          │
 operator ◀──── web (nginx + React) ──/api/ml──▶ ml-service (FastAPI, ONNX, DSP) ──────────┘          │
 console                │                                                                             │
                        └────────────/api/agent──▶ agent (LangGraph ReAct, 8 tools, MCP) ─────────────┘
                                                        │ OpenAI-compatible API
                                                        ▼
                                                  Ollama qwen2.5:7b (local) or hosted LLM
```

## Data flow
1. A sensor posts an I/Q capture (`/ingest/iq`) or a batch of pulse descriptor words (`/ingest/pdw`). The gateway validates
   the payload (sizes, finiteness, ranges), counts it, and publishes it to `iq-chunks` or `pdw-batches` keyed by sensor id,
   so a sensor's messages stay ordered inside one partition.
2. The consumer polls both topics in batches of up to 64. I/Q batches of equal length go through one ONNX call; the top class,
   probability, model latency and a Welch PSD are written to `detections` and `spectra`. PDW batches are deinterleaved
   (DBSCAN over standardised RF/PW/AOA, PRI statistics per cluster) into `resm` detections.
3. All detections of the batch feed the fuser once per sensor (`Fuser.step_multi`): predict, associate with the Hungarian
   algorithm on gated Mahalanobis distance, update the Kalman state [RF, AOA, RF rate, AOA rate], age unmatched tracks.
   The live Electronic Order of Battle is upserted into `tracks`.
4. The ML service serves inference and DSP endpoints and reads the tables for the console. The agent answers questions by
   calling tools over the same tables, the knowledge base and the ML service, and writes reports to `reports`.

## Architecture decision records
- **ADR-1 Java for ingest, Python for ML.** The JVM gateway validates and isolates untrusted sensor input (bean validation,
  size limits, typed records) and gives actuator health/metrics for free; the ML side iterates faster in Python.
- **ADR-2 Kafka rather than ZeroMQ.** Replayability: recorded topics become training sets and can be re-run through a new
  model. Per-sensor keys preserve ordering. ZeroMQ remains the option for edge-to-node links where sub-millisecond latency matters.
- **ADR-3 ONNX Runtime serving.** One exported graph (opset 17, dynamic batch and length axes) runs on CPU, CUDA or TensorRT
  execution providers without code change. Adaptive pooling makes the model length-agnostic, so 128-sample and 1024-sample
  captures share weights. Export parity against PyTorch is checked at export time and recorded in the results JSON.
- **ADR-4 Local embeddings and local LLM by default.** bge-small (fastembed, ONNX) for retrieval and Ollama for generation
  keep the platform air-gap compatible; a hosted OpenAI-compatible endpoint is a three-variable change in `.env`.
- **ADR-5 Prototypical few-shot instead of per-emitter fine-tuning.** A new modulation or emitter needs five labelled
  captures and no retraining: class prototypes are mean embeddings, classification is nearest centroid. Deterministic and
  auditable in the field.
- **ADR-6 Hybrid retrieval.** Dense cosine search alone blurs exact literals (channel numbers, frequencies). Postgres
  full-text and an exact numeric match are fused with reciprocal rank fusion; the eval reports both dense-only and hybrid so
  the effect is measured, not assumed.
- **ADR-7 Synthetic RadioML-layout data as the default training source.** DeepSig RadioML needs a licence and is not
  redistributable. `data/synth_mod.py` produces the same 24 classes with the same file layouts (RRC pulse shaping, CFO,
  sample-rate offset, multipath, AWGN), so the pipeline is runnable end to end by anyone. Every results JSON carries a
  `synthetic` flag, and the report generator labels the numbers accordingly.
- **ADR-8 Consumer batching sized to model latency.** `max_poll_records=64` with `max_poll_interval_ms=60000` keeps
  inference well inside the poll interval so heavy batches never trigger a consumer-group rebalance storm.
- **ADR-9 Edge holds no threat library.** Sensors send raw captures and PDWs; identification and the emitter catalogue live in
  the backend, so a catalogue update is a re-ingest, not a fleet software update.

## Operational notes
- Every container runs as a non-root user and has a health check; compose `depends_on` uses health conditions.
- The web container's nginx is the only public port; backends bind to loopback on the host.
- Prometheus metrics: `ml-service:8000/metrics` (request counters, inference histogram), `gateway:8080/actuator/prometheus`.
- Ollama on a host whose proxy breaks in-container pulls: `scripts/ollama_sideload.sh qwen2.5 7b` fetches the model with host
  curl, verifies SHA-256 and copies it into the volume.

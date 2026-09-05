# gateway-java

Spring Boot 3.3 / Java 21 ingest gateway. Validates edge-sensor payloads (I/Q chunks, PDW batches) and publishes them to Kafka, keyed by sensor so per-sensor ordering is preserved within a partition. The JVM gateway isolates untrusted sensor input from the Python ML service.

## Endpoints

| Method | Path | Body | Response |
|---|---|---|---|
| POST | `/ingest/iq` | `{"sensorId":"sirius-0","rfMhz":9400,"aoa":12.3,"i":[...],"q":[...]}` | `202 {"accepted":n,"samples":k}` |
| POST | `/ingest/pdw` | `{"sensorId":"sirius-esm","pdw":[[toa_ms,rf_mhz,pw_us,aoa_deg],...]}` | `202 {"accepted":n,"pulses":k}` |
| GET | `/ingest/stats` | - | `200 {"accepted_total","iq_total","pdw_total","rejected_total"}` |
| GET | `/actuator/health` | - | Spring health, includes the `kafka` component (200 when UP, 503 otherwise) |
| GET | `/actuator/health/readiness` | - | readiness group: `readinessState` + `kafka` |
| GET | `/actuator/health/liveness` | - | liveness group (does not depend on Kafka) |
| GET | `/actuator/metrics`, `/actuator/metrics/{name}` | - | Micrometer metrics |
| GET | `/actuator/prometheus` | - | Prometheus scrape endpoint |
| GET | `/actuator/info` | - | build info |

`accepted` in the 202 body is `accepted_total` after this request. The send to Kafka is asynchronous and not awaited; producer failures are logged, not returned.

Any validation failure, unreadable JSON, or `IllegalArgumentException` yields `400 {"error":"..."}` and increments `rejected_total`. A synchronous Kafka producer failure (broker unreachable for longer than `max.block.ms` = 5 s) yields `503 {"error":"kafka unavailable"}`.

## Validation rules

I/Q chunk (`/ingest/iq`):
- `sensorId`: required, 1..64 chars, `[A-Za-z0-9._-]+`
- `i`, `q`: required, 64..65536 samples each, same length, every value finite (JSON `null`, `NaN`, `Infinity` or values that overflow a double are rejected)
- `rfMhz`: optional; if present `0 < rfMhz < 1e6`
- `aoa`: optional; if present `0 <= aoa <= 360`

PDW batch (`/ingest/pdw`):
- `sensorId`: same rule as above
- `pdw`: required, 1..10000 rows, each row exactly 4 finite numbers `[toa_ms, rf_mhz, pw_us, aoa_deg]`

## Kafka

- Topics `iq-chunks` and `pdw-batches` are declared as `NewTopic` beans (3 partitions, replication 1) and created by `KafkaAdmin` at startup. `fail-fast` is off: if the broker is down at startup the gateway still starts and logs the error.
- Record key: `sensorId`. Values are JSON strings:
  - iq: `{"sensor_id","rf_mhz","aoa","i","q"}` (`rf_mhz`/`aoa` are `null` when omitted)
  - pdw: `{"sensor_id","pdw":[[...],...]}`
- Producer: `acks=1`, `compression.type=lz4`, `linger.ms=5`, `batch.size=65536`, `max.block.ms=5000`.
- Health: `KafkaHealthIndicator` lists topics through an `AdminClient` with a 2 s timeout and reports `UP` with the topic count (plus `missingTopics` if the two gateway topics are absent) or `DOWN` with the error.

## Metrics

Counters registered in the `MeterRegistry` (visible under `/actuator/metrics/<name>` and as `<name>_total` in `/actuator/prometheus`), and served as JSON by `/ingest/stats`:

| Meter | Prometheus | Stats key |
|---|---|---|
| `ingest.accepted` | `ingest_accepted_total` | `accepted_total` |
| `ingest.iq` | `ingest_iq_total` | `iq_total` |
| `ingest.pdw` | `ingest_pdw_total` | `pdw_total` |
| `ingest.rejected` | `ingest_rejected_total` | `rejected_total` |

Standard Spring/JVM/HTTP-server metrics are exposed as well.

## Configuration

| Env | Default | Purpose |
|---|---|---|
| `PORT` | `8080` | HTTP port (`server.port`) |
| `KAFKA` | `localhost:9092` | bootstrap servers |

Also set in `application.yml`: graceful shutdown (20 s per phase), virtual threads (`spring.threads.virtual.enabled=true`), Jackson `fail-on-null-for-primitives`, health details always shown, liveness/readiness probes enabled.

## Build and test (Docker only)

There is no Maven on the host and the host JDK is 17, so everything runs in the `maven:3.9-eclipse-temurin-21` image. The named volume `sigint-m2` caches dependencies between runs. Maven does not read the `*_proxy` env vars, so the proxy is passed as JVM properties:

```sh
cd gateway-java
docker run --rm -v "$PWD":/src -w /src -v sigint-m2:/root/.m2 \
  -e MAVEN_OPTS="-Dhttps.proxyHost=cias.geoaws.com -Dhttps.proxyPort=8080 -Dhttp.proxyHost=cias.geoaws.com -Dhttp.proxyPort=8080" \
  maven:3.9-eclipse-temurin-21 mvn -q -B package
```

`package` compiles, runs the JUnit 5 tests (`src/test/java/.../IngestControllerTest.java`, a `@WebMvcTest` with a mocked `KafkaTemplate`) and writes `target/gateway-0.1.0.jar`. Test reports land in `target/surefire-reports/`. Drop the proxy `MAVEN_OPTS` when not behind a proxy.

Container image (multi-stage, tests skipped in the build stage):

```sh
docker build --build-arg HTTPS_PROXY=$https_proxy --build-arg HTTP_PROXY=$http_proxy -t sigint-gateway:test .
docker run --rm -p 8080:8080 -e KAFKA=kafka:9092 sigint-gateway:test
```

The build args are optional; the Dockerfile turns them into `MAVEN_OPTS` proxy properties for the build stage. The runtime image runs as a non-root user and has a `HEALTHCHECK` that requests `/actuator/health` over bash's `/dev/tcp` (the JRE image has neither curl nor wget) and passes only on HTTP 200, i.e. when Kafka is reachable.

Manual smoke test:

```sh
curl -s -X POST localhost:8080/ingest/iq -H 'content-type: application/json' -d @iq.json   # iq.json shaped as in the table above, 64..65536 samples
curl -s localhost:8080/ingest/stats
curl -s localhost:8080/actuator/health
```

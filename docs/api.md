# HTTP API contract

All services are reached through the web container's nginx on port 3000. The browser only talks to same-origin paths, so there is no CORS in production:

| Prefix | Upstream | Notes |
|---|---|---|
| `/api/ml/` | `ml-service:8000/` | inference + dashboard feeds |
| `/api/agent/` | `agent:8001/` | analyst chat + reports |
| `/api/gateway/` | `gateway:8080/` | ingest stats + actuator |

In local dev (Vite on 5173) the same prefixes are proxied by `vite.config.ts` to localhost:8000 / 8001 / 8080.

## ml-service

`GET /health` -> `{status:"ok"|"degraded", model:string, model_loaded:bool, classes:number, providers:string[], uptime_s:number, version:string}`

`GET /detections?limit=50&minutes=10&sensor=` -> newest first
```json
[{"id":1,"ts":"2026-09-04T12:00:00.123+00:00","sensor_id":"sirius-0","rf_mhz":9400.0,"aoa":123.4,"modulation":"QPSK","confidence":0.93,"latency_ms":1.2,"source":"comint"}]
```
`source` is `comint` (I/Q classifier) or `resm` (PDW deinterleaver; then `modulation` is `pulse`).

`GET /sensors` -> `[{"sensor_id":"sirius-0","last_seen":ts,"detections_10m":123,"kind":"comint"|"resm"}]`

`GET /spectrum?sensor=sirius-0` -> latest PSD of that sensor (or the most recent of any sensor when omitted)
```json
{"sensor_id":"sirius-0","ts":ts,"rf_mhz":9400.0,"levels":[-31.2, ...256 floats dB...]}
```
`GET /spectrum/history?sensor=&n=60` -> `{"sensor_id":..., "rows":[{"ts":ts,"rf_mhz":..,"levels":[...]}]}` oldest first (waterfall).

`GET /eob` -> current Electronic Order of Battle (fused tracks), strongest first
```json
[{"track_id":3,"rf_mhz":9400.1,"aoa":123.0,"rf_rate":0.0,"aoa_rate":0.2,"sources":["comint","resm"],"sensors":["sirius-0","sirius-esm"],
  "hits":41,"misses":0,"first_seen":ts,"last_seen":ts,"modulation":"pulse","label_agreement":0.9,"pw_us":0.5,"pri_us":500.0,"pri_type":"constant","updated":ts}]
```

`GET /stats` -> `{"detections_total":n,"detections_10m":n,"per_modulation":{"QPSK":n},"per_sensor":{"sirius-0":n},"tracks":n,"latency_ms":{"p50":x,"p95":x,"p99":x},"low_confidence_10m":n}`

`POST /classify` `{"i":[...],"q":[...],"top_k":5}` (64..65536 samples) -> `{"predictions":[{"modulation":"QPSK","prob":0.91}],"latency_ms":1.3,"n_samples":1024}`

`POST /detect` `{"i":[...],"q":[...]}` -> `{"freqs":[...],"levels_db":[...],"mask":[bool...],"n_detections":k}` (full PSD plus CFAR mask, for plotting)

`POST /spectrogram` `{"i":[...],"q":[...]}` -> `{"freqs":[...],"times":[...],"db":[[...]]}` (rows = freq bins)

`POST /deinterleave` `{"pdw":[[toa_ms,rf_mhz,pw_us,aoa_deg],...]}` -> `{"labels":[...],"emitters":{"0":{"n":..,"rf":..,"pw":..,"aoa":..,"pri_mean":..,"pri_type":"constant"}}}`

`POST /synth` `{"modulation":"QPSK","snr_db":10,"length":1024,"seed":0}` -> `{"i":[...],"q":[...],"modulation":..,"snr_db":..}` (generate a demo signal server-side)

`GET /modulations` -> `["OOK","4ASK",...]` classes of the loaded model.

`GET /metrics` -> Prometheus text.

`GET /reports?limit=20` -> `[{"id":1,"ts":ts,"question":"...","report":"markdown"}]` (also served by the agent).

## agent

`POST /ask` `{"question":"...","thread":"ui"}` ->
```json
{"answer":"markdown","trace":[{"tool":"recent_detections","args":{"minutes":10},"result_preview":"...","seconds":0.12}],"seconds":4.2,"model":"qwen2.5:7b"}
```
`GET /reports?limit=20` -> as above. `GET /health` -> `{"status":"ok","llm":{"base_url":..,"model":..,"reachable":bool}}`

## gateway

`POST /ingest/iq` `{"sensorId":"sirius-0","rfMhz":9400,"aoa":12.3,"i":[...],"q":[...]}` -> 202 `{"accepted":n,"samples":k}`; 400 on validation error `{"error":"..."}`.
`POST /ingest/pdw` `{"sensorId":"sirius-esm","pdw":[[toa,rf,pw,aoa],...]}` -> 202 `{"accepted":n,"pulses":k}`
`GET /ingest/stats` -> `{"accepted_total":n,"iq_total":n,"pdw_total":n,"rejected_total":n}`
`GET /actuator/health` -> Spring health.

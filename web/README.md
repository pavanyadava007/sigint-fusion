# SIGINT-Fusion web console

Operator console for the SIGINT-Fusion platform: Vite + React 18 + TypeScript
(strict), react-router-dom v6, hand-drawn canvas charts for the waterfall,
bearing rose, constellation and spectrogram, recharts for bar/line charts.

## Pages

| Route | Purpose |
|---|---|
| `/` | Overview: stat tiles, spectrum waterfall, live detections, compact EOB rose |
| `/emitters` | Electronic Order of Battle: fused tracks table, bearing rose, track detail |
| `/analyst` | Analyst chat (`POST /api/agent/ask`), tool-call trace, saved reports drawer |
| `/lab` | Signal lab: synthesise or paste I/Q, constellation, PSD + CFAR, spectrogram, classifier top-5 |
| `/system` | Backend health, model info, latency percentiles, per-modulation and per-sensor counts |

All backend calls are same-origin under `/api/ml/`, `/api/agent/` and
`/api/gateway/` (contract: `../docs/api.md`). Every panel degrades honestly
when a backend is unreachable: it shows an "ml-service unreachable" style
status and holds the last good data at reduced opacity; nothing is faked.
Polling pauses while the browser tab is hidden.

## Development

```
npm install
npm run dev          # http://localhost:5173
```

`vite.config.ts` proxies `/api/ml` -> `http://localhost:8000`, `/api/agent` ->
`http://localhost:8001` and `/api/gateway` -> `http://localhost:8080`, stripping
the prefix. Start the backends locally (see the repository README) or leave
them down to exercise the unreachable states.

```
npm run typecheck    # tsc --noEmit
npm run lint         # eslint
npm run build        # typecheck + vite build -> dist/
npm run preview      # serve dist/ locally (no API proxy)
```

## Production image

```
docker build -t sigint-fusion-web .
docker run --rm -p 3000:80 sigint-fusion-web
```

The multi-stage `Dockerfile` builds with `node:22-alpine` and serves `dist/`
with `nginx:alpine` using `nginx.conf`, which provides the SPA fallback, gzip,
security headers and the three reverse proxies to the docker-compose service
names `ml-service:8000`, `agent:8001` and `gateway:8080` (the agent proxy has a
300 s read timeout for long LLM answers). Upstream names are resolved through
Docker's embedded DNS (`127.0.0.11`) at request time, so the container starts
even if an upstream is down; the console then reports that backend as
unreachable.

Validate the nginx configuration without building:

```
docker run --rm -v $PWD/nginx.conf:/etc/nginx/conf.d/default.conf:ro nginx:alpine nginx -t
```

## Notes on the API contract

- `docs/api.md` has no sample rate for spectrum sweeps, so the waterfall's
  frequency axis is labelled as bin offset relative to the sensor's centre
  frequency (`-fs/2 .. fc .. +fs/2`).
- `POST /api/agent/ask` fields `trace`, `seconds` and `model` are treated as
  optional; older agent builds return only `answer`.
- Detections below 0.60 confidence are tinted red in the tables.
- The analyst thread id is generated per browser tab and kept in
  `sessionStorage`; the chat transcript is kept there too.

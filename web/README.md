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

## Browser mode (static hosting, no backend)

`npm run build:browser` produces `dist-browser/`, a build of the same console
that runs entirely inside the browser tab, for static hosts such as a Hugging
Face static Space. The pages, components and `src/api.ts` are byte-for-byte
the same as in the server build; what changes is underneath them:

- `src/browser/shim.ts` wraps `window.fetch` and answers every path under
  `/api/ml/`, `/api/agent/` and `/api/gateway/` in-page with the field layout
  from `../docs/api.md`. Anything else passes through to the real `fetch`.
- `src/browser/model.ts` runs the ResNet-1D classifier (`public/demo/model.onnx`)
  with onnxruntime-web (wasm execution provider, single thread, main thread).
  Input `iq` float32 `[batch, 2, length]`, RMS-normalised like ml-service.
- `src/browser/dsp.ts` ports `ml-service/sigproc/dsp.py`: Welch PSD (256 bins),
  CA-CFAR (guard 2, train 8, Pfa 1e-3), STFT spectrogram (nperseg 64, hop 32)
  and DBSCAN PDW deinterleaving (eps 0.15, min_samples 5), all on a small
  radix-2 FFT.
- `src/browser/fusion.ts` ports `ml-service/fusion/associate.py`: Kalman
  tracks `[rf, aoa, rf_rate, aoa_rate]`, Mahalanobis gate 4, greedy
  nearest-within-gate assignment, duplicate merging per sensor group,
  `max_misses` 20.
- `src/browser/sim.ts` ports `ml-service/sensor_sim.py`: the 7-emitter
  scenario, three COMINT sensors (`sirius-0..2`) that classify real RadioML
  frames from `public/demo/frames.f32`, one R-ESM sensor (`sirius-esm`) that
  deinterleaves 40 ms PDW batches every second tick, all fused into the EOB.
  Four ticks per second, paused while the tab is hidden. In-memory tables:
  detections (5000), spectra per sensor (120), tracks, reports, gateway counters.
- `src/browser/analyst.ts` gathers the evidence in the tab (frequency
  extraction and emitter-catalogue lookup, BM25-style retrieval over
  `public/demo/corpus.json` with `[source]` tags, live-picture summary, EOB
  table) and sends it as compact JSON context (about 8 kB max) to
  `src/browser/llm.ts`: Qwen2.5-7B-Instruct served by a Gradio app on the
  Hugging Face ZeroGPU Space `pavanyadava07/sigint-fusion-agent` (named
  endpoint `/ask`, via `@gradio/client`). The client is created once and
  reused; a failed connect is remembered for 60 s. If the Space is asleep
  (30-90 s cold start), over its ZeroGPU quota, slower than 120 s or disabled,
  the same evidence is answered by rule-based templates instead, prefixed with
  an italic note, and the `model` field reads "rule-based analyst (fallback)".
  Reports (Summary / Emitters / Assessment / Confidence / Recommended actions)
  are saved to the in-memory reports table in both paths.
- `GET /api/agent/health` probes the Space at most once per 60 s
  (`Client.connect` + `view_api`, 15 s timeout) and reports
  `llm.reachable` plus a detail string, so the `llm` chip is green only when
  the Space is awake. `?llm=off` in the page URL (before the `#`) disables the
  backend, `?llm=owner/space` points at another Space, and `VITE_LLM_SPACE` sets
  the build-time default.
- `POST /api/ml/synth` returns a real RadioML frame of the requested
  modulation with white Gaussian noise added to reach the requested SNR
  relative to the frame's own power (`synthetic: false`); modulations not in
  the frame bank (8 of the 21 classes are available) answer HTTP 400.
- The router is a `HashRouter` (deep links such as `#/emitters` work on any
  static host) and `base` is `./`, so the build can live under a sub-path.
- A one-line strip under the header marks the build as a browser demo; the
  rest of the UI is identical.

Switching is done with `import.meta.env.MODE === 'browser'` in `src/main.tsx`
and `src/components/Layout.tsx`; `vite.config.ts` resolves
`virtual:sigint-browser` to `src/browser/boot.ts` in browser mode and to an
empty stub otherwise, so the normal build never bundles onnxruntime-web.

```
npm run build:browser                                 # -> dist-browser/
python3 -m http.server 5199 --directory dist-browser  # or: npx serve -l 5199 dist-browser
```

`dist-browser/` contains `index.html`, `assets/`, `demo/` (model, frames,
corpus, model card) and `ort-wasm-simd-threaded.{wasm,mjs}` copied from
onnxruntime-web next to `index.html` (`ort.env.wasm.wasmPaths` resolves `./`
against the page URL). Upload the whole directory to the static host.

Demo assets in `public/demo/`:

| File | Content |
|---|---|
| `model.onnx`, `classes.json` | ResNet-1D classifier (ONNX opset 17) and its 21 class names in logit order |
| `frames.f32`, `frames.json` | 320 real RadioML 2018.01A frames, 8 modulations, `[n][2][1024]` float32 little-endian, RMS-normalised |
| `corpus.json` | knowledge-base chunks and the emitter catalogue used by the analyst |
| `model_card.json` | training data, validation accuracy vs SNR, device |

## Notes on the API contract

- `docs/api.md` has no sample rate for spectrum sweeps, so the waterfall's
  frequency axis is labelled as bin offset relative to the sensor's centre
  frequency (`-fs/2 .. fc .. +fs/2`).
- `POST /api/agent/ask` fields `trace`, `seconds` and `model` are treated as
  optional; older agent builds return only `answer`.
- Detections below 0.60 confidence are tinted red in the tables.
- The analyst thread id is generated per browser tab and kept in
  `sessionStorage`; the chat transcript is kept there too.

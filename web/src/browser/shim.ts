// window.fetch wrapper for browser mode. Every path under /api/ml/, /api/agent/ and /api/gateway/
// is answered in-page from the simulation tables (contract: docs/api.md); everything else passes
// through to the real fetch. The pages and src/api.ts are untouched.

import { Analyst } from './analyst';
import { BACKEND, DEFAULT_MODEL, probeLlm } from './llm';
import { nativeFetch } from './bank';
import { caCfar, deinterleavePdw, round, stft, welchPsd } from './dsp';
import { Rng } from './rng';
import type { Store } from './sim';

export const VERSION = '1.0.0-browser';
export const MODEL_NAME = 'demo/model.onnx';

class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

function text(body: string, status = 200): Response {
  return new Response(body, { status, headers: { 'content-type': 'text/plain; version=0.0.4; charset=utf-8' } });
}

function intParam(p: URLSearchParams, name: string, def: number, lo: number, hi: number): number {
  const raw = p.get(name);
  if (raw === null || raw === '') return def;
  const v = Number(raw);
  if (!Number.isFinite(v)) throw new HttpError(422, `query parameter ${name} must be a number`);
  return Math.max(lo, Math.min(hi, Math.trunc(v)));
}

/** Drop the internal epoch-ms field before a row leaves the shim. */
function withoutT<T extends { t: number }>(row: T): Omit<T, 't'> {
  const copy: Partial<T> = { ...row };
  delete copy.t;
  return copy as Omit<T, 't'>;
}

function percentile(sorted: number[], p: number): number | null {
  if (!sorted.length) return null;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

interface IqBody {
  i?: unknown;
  q?: unknown;
  top_k?: unknown;
}

function parseIq(body: unknown): { i: number[]; q: number[]; topK: number } {
  const b = (body ?? {}) as IqBody;
  if (!Array.isArray(b.i) || !Array.isArray(b.q)) throw new HttpError(422, 'body must contain numeric arrays "i" and "q"');
  if (b.i.length < 64 || b.i.length > 65536 || b.q.length < 64 || b.q.length > 65536) throw new HttpError(422, 'i and q must have between 64 and 65536 samples');
  if (b.i.length !== b.q.length) throw new HttpError(422, 'i/q length mismatch');
  const i = (b.i as unknown[]).map(Number);
  const q = (b.q as unknown[]).map(Number);
  if (!i.every(Number.isFinite) || !q.every(Number.isFinite)) throw new HttpError(400, 'non-finite sample values');
  const topK = b.top_k === undefined ? 5 : Number(b.top_k);
  if (!Number.isInteger(topK) || topK < 1 || topK > 24) throw new HttpError(422, 'top_k must be an integer between 1 and 24');
  return { i, q, topK };
}

export class Shim {
  private readonly analyst: Analyst;
  constructor(private readonly store: Store) {
    this.analyst = new Analyst(store);
  }

  // ---------- ml-service ----------

  private mlGet(path: string, p: URLSearchParams): Response {
    const s = this.store;
    switch (path) {
      case 'health':
        return json({
          status: 'ok',
          model: MODEL_NAME,
          model_loaded: true,
          error: null,
          classes: s.clf.classes.length,
          providers: ['wasm'],
          uptime_s: s.uptimeS,
          version: VERSION,
        });
      case 'modulations':
        return json(s.clf.classes);
      case 'detections': {
        const limit = intParam(p, 'limit', 50, 1, 1000);
        const minutes = intParam(p, 'minutes', 60, 1, 100000);
        const sensor = p.get('sensor') || undefined;
        return json(s.recentDetections(minutes, limit, sensor).map(withoutT));
      }
      case 'sensors': {
        const since10 = Date.now() - 600_000;
        const since1d = Date.now() - 86_400_000;
        const bySensor = new Map<string, { last: number; n10: number; kinds: Map<string, number> }>();
        for (const d of s.detections) {
          if (d.t <= since1d) continue;
          const e = bySensor.get(d.sensor_id) ?? { last: 0, n10: 0, kinds: new Map() };
          e.last = Math.max(e.last, d.t);
          if (d.t > since10) e.n10 += 1;
          e.kinds.set(d.source, (e.kinds.get(d.source) ?? 0) + 1);
          bySensor.set(d.sensor_id, e);
        }
        return json(
          Array.from(bySensor.entries())
            .sort((a, b) => a[0].localeCompare(b[0]))
            .map(([sensor_id, e]) => ({
              sensor_id,
              last_seen: new Date(e.last).toISOString(),
              detections_10m: e.n10,
              kind: Array.from(e.kinds.entries()).sort((a, b) => b[1] - a[1])[0][0],
            })),
        );
      }
      case 'spectrum': {
        const sensor = p.get('sensor') || undefined;
        const latest = this.latestSpectrum(sensor);
        if (!latest) throw new HttpError(404, 'no spectra yet');
        return json({ sensor_id: latest.sensor_id, ...withoutT(latest.row) });
      }
      case 'spectrum/history': {
        const n = intParam(p, 'n', 60, 1, 500);
        let sensor = p.get('sensor') || undefined;
        if (!sensor) {
          const latest = this.latestSpectrum();
          if (!latest) return json({ sensor_id: null, rows: [] });
          sensor = latest.sensor_id;
        }
        const rows = (s.spectra.get(sensor) ?? []).slice(-n).map(withoutT);
        return json({ sensor_id: sensor, rows });
      }
      case 'eob':
        return json(s.tracks);
      case 'stats': {
        const recent = s.recentDetections(10, 1_000_000);
        const perMod: Record<string, number> = {};
        const perSensor: Record<string, number> = {};
        const lat: number[] = [];
        let low = 0;
        for (const d of recent) {
          perMod[d.modulation] = (perMod[d.modulation] ?? 0) + 1;
          perSensor[d.sensor_id] = (perSensor[d.sensor_id] ?? 0) + 1;
          if (d.confidence < 0.6) low += 1;
          lat.push(d.latency_ms);
        }
        lat.sort((a, b) => a - b);
        const pct = (q: number) => {
          const v = percentile(lat, q);
          return v === null ? null : round(v, 3);
        };
        return json({
          detections_total: s.detections.length,
          detections_10m: recent.length,
          low_confidence_10m: low,
          per_modulation: Object.fromEntries(Object.entries(perMod).sort((a, b) => b[1] - a[1])),
          per_sensor: Object.fromEntries(Object.entries(perSensor).sort((a, b) => a[0].localeCompare(b[0]))),
          tracks: s.tracks.length,
          latency_ms: { p50: pct(0.5), p95: pct(0.95), p99: pct(0.99) },
        });
      }
      case 'reports':
        return this.reports(p);
      case 'metrics':
        return text(
          [
            '# HELP ml_requests_total requests (browser demo placeholder)',
            '# TYPE ml_requests_total counter',
            `ml_requests_total{endpoint="classify",status="200"} ${s.counters.iq_total}`,
            `ml_requests_total{endpoint="deinterleave",status="200"} ${s.counters.pdw_total}`,
            '# HELP ml_uptime_seconds seconds since the browser demo started',
            '# TYPE ml_uptime_seconds gauge',
            `ml_uptime_seconds ${s.uptimeS}`,
            '',
          ].join('\n'),
        );
      default:
        throw new HttpError(404, 'Not Found');
    }
  }

  private latestSpectrum(sensor?: string): { sensor_id: string; row: NonNullable<ReturnType<Store['spectra']['get']>>[number] } | null {
    let best: { sensor_id: string; row: NonNullable<ReturnType<Store['spectra']['get']>>[number] } | null = null;
    for (const [sid, rows] of this.store.spectra) {
      if (sensor && sid !== sensor) continue;
      const row = rows[rows.length - 1];
      if (row && (!best || row.t > best.row.t)) best = { sensor_id: sid, row };
    }
    return best;
  }

  private reports(p: URLSearchParams): Response {
    const limit = intParam(p, 'limit', 20, 1, 200);
    return json(this.store.reports.slice(-limit).reverse());
  }

  private async mlPost(path: string, body: unknown): Promise<Response> {
    const s = this.store;
    switch (path) {
      case 'classify': {
        const { i, q, topK } = parseIq(body);
        const { probs, latencyMs } = await s.clf.classify(i, q);
        const order = probs.map((_pr, k) => k).sort((a, b) => probs[b] - probs[a]).slice(0, topK);
        return json({ predictions: order.map((k) => ({ modulation: s.clf.classes[k], prob: round(probs[k], 4) })), latency_ms: round(latencyMs, 2), n_samples: i.length });
      }
      case 'detect': {
        const { i, q } = parseIq(body);
        const psd = welchPsd(i, q);
        const mask = caCfar(psd.db);
        return json({ freqs: psd.freqs.map((v) => round(v, 4)), levels_db: psd.db.map((v) => round(v, 2)), mask, n_detections: mask.filter(Boolean).length });
      }
      case 'spectrogram': {
        const { i, q } = parseIq(body);
        const r = stft(i, q);
        return json({ freqs: r.freqs.map((v) => round(v, 4)), times: r.times.map((v) => round(v, 2)), db: r.db.map((row) => row.map((v) => round(v, 1))) });
      }
      case 'deinterleave': {
        const pdw = (body as { pdw?: unknown } | null)?.pdw;
        if (!Array.isArray(pdw) || pdw.length < 1 || pdw.length > 20000) throw new HttpError(422, 'pdw must contain 1 to 20000 rows');
        if (!pdw.every((r) => Array.isArray(r) && r.length === 4 && r.every((v) => Number.isFinite(Number(v))))) throw new HttpError(400, 'pdw rows must be [toa_ms, rf_mhz, pw_us, aoa_deg]');
        const { labels, emitters } = deinterleavePdw((pdw as unknown[][]).map((r) => r.map(Number)));
        return json({ labels, emitters });
      }
      case 'synth':
        return this.synth(body);
      default:
        throw new HttpError(404, 'Not Found');
    }
  }

  /** Browser mode: a real RadioML frame of the requested modulation plus white Gaussian noise
   *  scaled to the requested SNR relative to the frame's own power. */
  private synth(body: unknown): Response {
    const b = (body ?? {}) as { modulation?: unknown; snr_db?: unknown; length?: unknown; seed?: unknown };
    const modulation = typeof b.modulation === 'string' ? b.modulation : 'QPSK';
    const snr = b.snr_db === undefined ? 10 : Number(b.snr_db);
    const length = b.length === undefined ? 1024 : Number(b.length);
    const seed = b.seed === undefined ? 0 : Number(b.seed);
    if (!Number.isFinite(snr) || snr < -20 || snr > 30) throw new HttpError(422, 'snr_db must be between -20 and 30');
    if (!Number.isInteger(length) || length < 64 || length > 8192) throw new HttpError(422, 'length must be an integer between 64 and 8192');
    const bank = this.store.assets.bank;
    if (!bank.has(modulation)) throw new HttpError(400, `no real frames for ${modulation} in the browser demo; choose one of ${JSON.stringify(bank.modulations)}`);
    const rng = new Rng(Number.isFinite(seed) ? seed : 0);
    const L = bank.meta.length;
    const i: number[] = [];
    const q: number[] = [];
    const frames: number[] = [];
    while (i.length < length) {
      const fr = bank.draw(modulation, rng);
      frames.push(fr.index);
      const take = Math.min(L, length - i.length);
      const off = take < L ? rng.int(L - take + 1) : 0; // a random window when a shorter signal is requested
      for (let k = 0; k < take; k++) {
        i.push(fr.i[off + k]);
        q.push(fr.q[off + k]);
      }
    }
    let ps = 0;
    for (let k = 0; k < i.length; k++) ps += i[k] * i[k] + q[k] * q[k];
    ps /= i.length;
    const pn = ps / Math.pow(10, snr / 10);
    const sd = Math.sqrt(pn / 2);
    for (let k = 0; k < i.length; k++) {
      i[k] = round(i[k] + rng.normal(0, sd), 5);
      q[k] = round(q[k] + rng.normal(0, sd), 5);
    }
    return json({
      i,
      q,
      modulation,
      snr_db: snr,
      synthetic: false,
      note: 'real RadioML frame + added noise',
      frames,
      frame_snr_db: frames.map((idx) => bank.meta.snr_db[idx]),
    });
  }

  // ---------- agent ----------

  private async agentGet(path: string, p: URLSearchParams): Promise<Response> {
    switch (path) {
      case 'health': {
        const st = await probeLlm();
        return json({
          status: 'ok',
          llm: st
            ? { base_url: BACKEND, model: st.model, reachable: st.reachable, detail: st.detail }
            : { base_url: BACKEND, model: DEFAULT_MODEL, detail: 'probing the ZeroGPU Space; rule-based fallback active meanwhile' },
        });
      }
      case 'reports':
        return this.reports(p);
      default:
        throw new HttpError(404, 'Not Found');
    }
  }

  private async agentPost(path: string, body: unknown): Promise<Response> {
    if (path !== 'ask') throw new HttpError(404, 'Not Found');
    const b = (body ?? {}) as { question?: unknown; thread?: unknown };
    if (typeof b.question !== 'string' || !b.question.trim()) throw new HttpError(422, 'question must be a non-empty string');
    const thread = typeof b.thread === 'string' && b.thread ? b.thread : 'ui';
    return json(await this.analyst.ask(b.question, thread));
  }

  // ---------- gateway ----------

  private gatewayGet(path: string): Response {
    switch (path) {
      case 'actuator/health':
        return json({ status: 'UP', components: { kafka: { status: 'UP', details: { topics: 2 } } } });
      case 'ingest/stats':
        return json({ ...this.store.counters });
      default:
        throw new HttpError(404, 'Not Found');
    }
  }

  // ---------- dispatcher ----------

  async handle(method: string, service: string, path: string, params: URLSearchParams, body: unknown): Promise<Response> {
    try {
      if (service === 'ml') return method === 'POST' ? await this.mlPost(path, body) : this.mlGet(path, params);
      if (service === 'agent') return method === 'POST' ? await this.agentPost(path, body) : await this.agentGet(path, params);
      if (service === 'gateway') return this.gatewayGet(path);
      throw new HttpError(404, 'Not Found');
    } catch (e) {
      if (e instanceof HttpError) return json({ detail: e.message }, e.status);
      const msg = e instanceof Error ? e.message : String(e);
      console.error('browser shim error', service, path, e);
      return json({ detail: msg }, 500);
    }
  }
}

const API_RE = /\/api\/(ml|agent|gateway)\/(.+?)\/?$/;

/**
 * Install the fetch wrapper. `ready` resolves to the Shim once the model and frame bank are loaded;
 * requests arriving earlier wait for it (or get a 503 if loading failed). /api/ml/health answers
 * immediately with a degraded status while loading so the health chips are honest.
 */
export function installFetchShim(ready: Promise<Shim>, t0: number): void {
  let shim: Shim | null = null;
  let loadError: string | null = null;
  ready.then(
    (s) => {
      shim = s;
    },
    (e) => {
      loadError = e instanceof Error ? e.message : String(e);
    },
  );
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    let u: URL;
    try {
      u = new URL(url, document.baseURI);
    } catch {
      return nativeFetch(input, init);
    }
    const m = u.origin === location.origin ? API_RE.exec(u.pathname) : null;
    if (!m) return nativeFetch(input, init);
    const service = m[1];
    const path = m[2];
    const method = (init?.method ?? (typeof input === 'object' && 'method' in input ? input.method : 'GET')).toUpperCase();
    if (!shim) {
      if (service === 'ml' && path === 'health' && method === 'GET') {
        return json({
          status: 'degraded',
          model: MODEL_NAME,
          model_loaded: false,
          error: loadError ?? 'loading model and frame bank',
          classes: 0,
          providers: [],
          uptime_s: round((Date.now() - t0) / 1000, 1),
          version: VERSION,
        });
      }
      if (loadError) return json({ detail: `browser demo failed to load: ${loadError}` }, 503);
      try {
        shim = await ready;
      } catch (e) {
        return json({ detail: `browser demo failed to load: ${e instanceof Error ? e.message : String(e)}` }, 503);
      }
    }
    let body: unknown = undefined;
    if (init?.body !== undefined && init.body !== null) {
      if (typeof init.body !== 'string') return json({ detail: 'only JSON string bodies are supported' }, 415);
      try {
        body = JSON.parse(init.body);
      } catch {
        return json({ detail: 'invalid JSON body' }, 400);
      }
    }
    // yield once so a burst of polls never starves rendering
    await new Promise<void>((r) => window.setTimeout(r, 0));
    return shim.handle(method, service, path, u.searchParams, body);
  };
}

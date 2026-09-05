// Typed client for the SIGINT-Fusion backends. All paths are same-origin;
// nginx (prod) or vite (dev) proxies /api/ml, /api/agent and /api/gateway.
// Contract: docs/api.md.

export type ErrorKind = 'network' | 'timeout' | 'http' | 'parse';

export class ApiError extends Error {
  readonly kind: ErrorKind;
  readonly status?: number;
  constructor(kind: ErrorKind, message: string, status?: number) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
  }
}

export function toApiError(e: unknown): ApiError {
  if (e instanceof ApiError) return e;
  if (e instanceof Error) return new ApiError('network', e.message);
  return new ApiError('network', String(e));
}

/** Short operator-facing description, e.g. "ml-service unreachable (HTTP 502)". */
export function describeError(service: string, e: ApiError | null | undefined): string {
  if (!e) return `${service} unreachable`;
  switch (e.kind) {
    case 'timeout':
      return `${service} timed out (${e.message})`;
    case 'http':
      if (e.status === 502 || e.status === 503 || e.status === 504) return `${service} unreachable (HTTP ${e.status})`;
      return `${service} error (${e.message})`;
    case 'parse':
      return `${service} returned an unreadable response`;
    default:
      return `${service} unreachable (${e.message})`;
  }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 8000): Promise<T> {
  const ctrl = new AbortController();
  const timer = window.setTimeout(() => ctrl.abort(), timeoutMs);
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      signal: ctrl.signal,
      headers: {
        accept: 'application/json',
        ...(init.body ? { 'content-type': 'application/json' } : {}),
        ...(init.headers ?? {}),
      },
    });
  } catch (e) {
    window.clearTimeout(timer);
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new ApiError('timeout', `no reply within ${Math.round(timeoutMs / 1000)} s`);
    }
    throw new ApiError('network', 'connection failed');
  }
  window.clearTimeout(timer);
  if (!res.ok) {
    let detail = '';
    try {
      const text = await res.text();
      // nginx 502 pages are HTML; only surface short, non-markup bodies.
      if (text && !text.trim().startsWith('<')) detail = text.trim().slice(0, 160);
    } catch {
      /* body not readable */
    }
    throw new ApiError('http', detail ? `HTTP ${res.status}: ${detail}` : `HTTP ${res.status}`, res.status);
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new ApiError('parse', 'invalid JSON');
  }
}

const get = <T>(path: string, timeoutMs?: number) => request<T>(path, {}, timeoutMs);
const post = <T>(path: string, body: unknown, timeoutMs?: number) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) }, timeoutMs);

function qs(params: Record<string, string | number | undefined>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== '') p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : '';
}

// ---------- ml-service types ----------

export type Source = 'comint' | 'resm';

export interface MlHealth {
  status: 'ok' | 'degraded' | string;
  model: string;
  model_loaded: boolean;
  classes: number;
  providers: string[];
  uptime_s: number;
  version: string;
}

export interface Detection {
  id: number;
  ts: string;
  sensor_id: string;
  rf_mhz: number;
  aoa: number | null;
  modulation: string;
  confidence: number;
  latency_ms: number;
  source: Source;
}

export interface Sensor {
  sensor_id: string;
  last_seen: string;
  detections_10m: number;
  kind: Source;
}

export interface SpectrumRow {
  ts: string;
  rf_mhz: number;
  levels: number[];
}

export interface Spectrum extends SpectrumRow {
  sensor_id: string;
}

export interface SpectrumHistory {
  sensor_id: string;
  rows: SpectrumRow[];
}

export interface Track {
  track_id: number;
  rf_mhz: number;
  aoa: number;
  rf_rate: number;
  aoa_rate: number;
  sources: Source[];
  sensors: string[];
  hits: number;
  misses: number;
  first_seen: string;
  last_seen: string;
  modulation: string;
  label_agreement: number;
  pw_us: number | null;
  pri_us: number | null;
  pri_type: string | null;
  updated: string;
}

export interface Stats {
  detections_total: number;
  detections_10m: number;
  per_modulation: Record<string, number>;
  per_sensor: Record<string, number>;
  tracks: number;
  latency_ms: { p50: number; p95: number; p99: number };
  low_confidence_10m: number;
}

export interface IQ {
  i: number[];
  q: number[];
}

export interface Prediction {
  modulation: string;
  prob: number;
}

export interface ClassifyResult {
  predictions: Prediction[];
  latency_ms: number;
  n_samples: number;
}

export interface DetectResult {
  freqs: number[];
  levels_db: number[];
  mask: boolean[];
  n_detections: number;
}

export interface SpectrogramResult {
  freqs: number[];
  times: number[];
  db: number[][]; // rows = freq bins, columns = time frames
}

export interface SynthRequest {
  modulation: string;
  snr_db: number;
  length: number;
  seed: number;
}

export interface SynthResult extends IQ {
  modulation: string;
  snr_db: number;
}

export interface Report {
  id: number;
  ts: string;
  question: string;
  report: string;
}

// ---------- agent types ----------

export interface TraceStep {
  tool: string;
  args: Record<string, unknown>;
  result_preview: string;
  seconds: number;
}

export interface AskResult {
  answer: string;
  // Optional: older agent builds return only { answer }.
  trace?: TraceStep[];
  seconds?: number;
  model?: string;
}

export interface AgentHealth {
  status: string;
  llm?: { base_url?: string; model?: string; reachable?: boolean };
}

// ---------- gateway types ----------

export interface GatewayHealth {
  status: string; // UP | DOWN | OUT_OF_SERVICE | UNKNOWN
  components?: Record<string, { status: string; details?: Record<string, unknown> }>;
}

export interface IngestStats {
  accepted_total: number;
  iq_total: number;
  pdw_total: number;
  rejected_total: number;
}

// ---------- clients ----------

export const ml = {
  health: () => get<MlHealth>('/api/ml/health', 5000),
  detections: (p: { limit?: number; minutes?: number; sensor?: string } = {}) =>
    get<Detection[]>(`/api/ml/detections${qs(p)}`),
  sensors: () => get<Sensor[]>('/api/ml/sensors'),
  spectrum: (sensor?: string) => get<Spectrum>(`/api/ml/spectrum${qs({ sensor })}`),
  spectrumHistory: (p: { sensor?: string; n?: number } = {}) =>
    get<SpectrumHistory>(`/api/ml/spectrum/history${qs(p)}`),
  eob: () => get<Track[]>('/api/ml/eob'),
  stats: () => get<Stats>('/api/ml/stats'),
  modulations: () => get<string[]>('/api/ml/modulations'),
  classify: (iq: IQ, topK = 5) =>
    post<ClassifyResult>('/api/ml/classify', { i: iq.i, q: iq.q, top_k: topK }, 30_000),
  detect: (iq: IQ) => post<DetectResult>('/api/ml/detect', { i: iq.i, q: iq.q }, 30_000),
  spectrogram: (iq: IQ) => post<SpectrogramResult>('/api/ml/spectrogram', { i: iq.i, q: iq.q }, 30_000),
  synth: (req: SynthRequest) => post<SynthResult>('/api/ml/synth', req, 30_000),
  reports: (limit = 20) => get<Report[]>(`/api/ml/reports${qs({ limit })}`),
};

export const agent = {
  health: () => get<AgentHealth>('/api/agent/health', 5000),
  ask: (question: string, thread: string) =>
    post<AskResult>('/api/agent/ask', { question, thread }, 300_000),
  reports: (limit = 20) => get<Report[]>(`/api/agent/reports${qs({ limit })}`),
};

export const gateway = {
  health: () => get<GatewayHealth>('/api/gateway/actuator/health', 5000),
  ingestStats: () => get<IngestStats>('/api/gateway/ingest/stats'),
};

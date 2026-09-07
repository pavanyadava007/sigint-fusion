// LLM backend for the browser demo: a Gradio app on a Hugging Face ZeroGPU Space with one named
// endpoint "/ask" (question, context, report) -> JSON string {answer, model, seconds, gpu, zerogpu}.
// The client is created lazily once and reused; a failed connect is remembered for 60 s so the
// page never hammers a sleeping Space. Health probes are cached for 60 s as well.

import { Client, type SpaceStatus } from '@gradio/client';

export const DEFAULT_SPACE = 'pavanyadava07/sigint-fusion-agent';
export const DEFAULT_MODEL = 'Qwen/Qwen2.5-7B-Instruct';
const RETRY_AFTER_MS = 60_000;
const PROBE_TTL_MS = 60_000;
const PROBE_TIMEOUT_MS = 15_000;
export const ASK_TIMEOUT_MS = 120_000;

/** Space name: `?llm=off` disables the backend, `?llm=owner/space` overrides it, else VITE_LLM_SPACE or the default. */
export function resolveSpace(): string | null {
  try {
    const q = new URLSearchParams(window.location.search).get('llm');
    if (q !== null) {
      const v = q.trim();
      if (!v || v === 'off' || v === '0' || v === 'none' || v === 'false') return null;
      return v;
    }
  } catch {
    /* no window */
  }
  const env = (import.meta.env.VITE_LLM_SPACE as string | undefined)?.trim();
  if (env === 'off') return null;
  return env || DEFAULT_SPACE;
}

export const SPACE = resolveSpace();
export const BACKEND = SPACE ? `hf-space:${SPACE}` : 'disabled';

/**
 * Direct URL of a Space (`owner/name` -> `https://owner-name.hf.space`). Connecting by URL lets the
 * client skip the huggingface.co host lookup, which is fronted by a bot-control WAF that can answer
 * browsers with a captcha page; the Space origin itself sends proper CORS headers.
 */
export function spaceUrl(space: string): string {
  if (/^https?:\/\//i.test(space)) return space.replace(/\/+$/, '');
  return `https://${space.toLowerCase().replace(/[^a-z0-9-]+/g, '-')}.hf.space`;
}

export interface LlmAnswer {
  answer: string;
  model: string;
  seconds: number;
  gpu?: boolean;
  zerogpu?: boolean;
}

export interface LlmStatus {
  reachable: boolean;
  detail: string;
  model: string;
  at: number;
}

function withTimeout<T>(p: Promise<T>, ms: number, what: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(`${what} timed out after ${Math.round(ms / 1000)} s`)), ms);
    p.then(
      (v) => {
        window.clearTimeout(timer);
        resolve(v);
      },
      (e) => {
        window.clearTimeout(timer);
        reject(e);
      },
    );
  });
}

/** Short operator-facing reason from a client error or Space status. */
export function shortReason(e: unknown): string {
  const msg = e instanceof Error ? e.message : typeof e === 'string' ? e : JSON.stringify(e);
  const m = (msg || 'unknown error').replace(/\s+/g, ' ').trim();
  if (/quota|zerogpu/i.test(m) && /exceed|limit|quota/i.test(m)) return 'ZeroGPU quota exceeded';
  if (/sleep/i.test(m)) return 'Space asleep';
  if (/not.?found|404/i.test(m)) return 'Space not found';
  if (/timed out/i.test(m)) return m;
  if (/fetch|network|failed to fetch|cors/i.test(m)) return 'network error';
  return m.length > 90 ? `${m.slice(0, 87)}...` : m;
}

let clientPromise: Promise<Client> | null = null;
let lastFailure: { at: number; reason: string } | null = null;
let lastSpaceStatus: SpaceStatus | null = null;

function connect(space: string): Promise<Client> {
  if (clientPromise) return clientPromise;
  if (lastFailure && Date.now() - lastFailure.at < RETRY_AFTER_MS) {
    const wait = Math.ceil((RETRY_AFTER_MS - (Date.now() - lastFailure.at)) / 1000);
    return Promise.reject(new Error(`${lastFailure.reason}; retry in ${wait} s`));
  }
  const opts = {
    status_callback: (s: SpaceStatus) => {
      lastSpaceStatus = s;
    },
  };
  // Direct Space origin first, without status_callback (with it the client first asks huggingface.co
  // for the runtime status). The name-based lookup with status_callback is only the second try: it
  // goes through huggingface.co but yields the sleeping / building status message for the operator.
  clientPromise = Client.connect(spaceUrl(space))
    .catch((direct: unknown) => Client.connect(space, opts).catch(() => Promise.reject(direct)))
    .catch((e: unknown) => {
    clientPromise = null;
    const status = lastSpaceStatus;
    const reason = status && status.status !== 'running' ? `Space ${status.status}${status.message ? ` (${status.message})` : ''}` : shortReason(e);
    lastFailure = { at: Date.now(), reason };
    throw new Error(reason);
  });
  return clientPromise;
}

/** Ask the backend. Rejects on timeout, disabled backend, connect failure or a bad payload. */
export async function askLlm(question: string, context: string, report: boolean, timeoutMs = ASK_TIMEOUT_MS): Promise<LlmAnswer> {
  if (!SPACE) throw new Error('LLM backend disabled (?llm=off)');
  const t0 = performance.now();
  const client = await withTimeout(connect(SPACE), timeoutMs, 'connect');
  const remaining = Math.max(1000, timeoutMs - (performance.now() - t0));
  const result = await withTimeout(client.predict('/ask', { question, context, report }), remaining, 'LLM call');
  const raw = Array.isArray(result.data) ? result.data[0] : result.data;
  let parsed: Partial<LlmAnswer>;
  try {
    parsed = typeof raw === 'string' ? (JSON.parse(raw) as Partial<LlmAnswer>) : (raw as Partial<LlmAnswer>);
  } catch {
    throw new Error('backend returned invalid JSON');
  }
  if (!parsed || typeof parsed.answer !== 'string' || !parsed.answer.trim()) throw new Error('backend returned no answer');
  return {
    answer: parsed.answer,
    model: typeof parsed.model === 'string' && parsed.model ? parsed.model : DEFAULT_MODEL,
    seconds: typeof parsed.seconds === 'number' ? parsed.seconds : (performance.now() - t0) / 1000,
    gpu: typeof parsed.gpu === 'boolean' ? parsed.gpu : undefined,
    zerogpu: typeof parsed.zerogpu === 'boolean' ? parsed.zerogpu : undefined,
  };
}

let probe: { at: number; result: LlmStatus | null; pending: Promise<LlmStatus> | null } = { at: 0, result: null, pending: null };

async function runProbe(): Promise<LlmStatus> {
  const at = Date.now();
  if (!SPACE) return { reachable: false, detail: 'disabled by ?llm=off; rule-based fallback active', model: DEFAULT_MODEL, at };
  try {
    const client = await withTimeout(connect(SPACE), PROBE_TIMEOUT_MS, 'connect');
    const api = await withTimeout(client.view_api(), PROBE_TIMEOUT_MS, 'view_api');
    const endpoints = Object.keys((api as { named_endpoints?: Record<string, unknown> }).named_endpoints ?? {});
    if (!endpoints.includes('/ask')) return { reachable: false, detail: `asleep or unreachable: Space has no /ask endpoint (${endpoints.join(', ') || 'none'}); rule-based fallback active`, model: DEFAULT_MODEL, at };
    return { reachable: true, detail: 'ZeroGPU Space awake', model: DEFAULT_MODEL, at };
  } catch (e) {
    return { reachable: false, detail: `asleep or unreachable: ${shortReason(e)}; rule-based fallback active`, model: DEFAULT_MODEL, at };
  }
}

/**
 * Backend status for GET /api/agent/health, probed at most once per 60 s. Returns the cached
 * result when fresh; otherwise starts a probe and waits up to `waitMs` for it (null while pending).
 */
export async function probeLlm(waitMs = 4000): Promise<LlmStatus | null> {
  const fresh = probe.result && Date.now() - probe.at < PROBE_TTL_MS;
  if (fresh) return probe.result;
  if (!probe.pending) {
    probe.at = Date.now();
    probe.pending = runProbe().then((r) => {
      probe = { at: Date.now(), result: r, pending: null };
      return r;
    });
  }
  const pending = probe.pending;
  const timeout = new Promise<null>((r) => window.setTimeout(() => r(null), waitMs));
  const first = await Promise.race([pending, timeout]);
  return first ?? probe.result;
}

import { useCallback, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  ml,
  toApiError,
  type ApiError,
  type ClassifyResult,
  type DetectResult,
  type IQ,
  type SpectrogramResult,
} from '../api';
import { Constellation } from '../components/Constellation';
import { EmptyState, Panel } from '../components/Panel';
import { SpectrogramCanvas } from '../components/SpectrogramCanvas';
import { usePoll } from '../hooks/usePoll';
import { useThemeVersion } from '../hooks/useTheme';
import { fmtMs, fmtNum, fmtPct } from '../lib/format';
import { chartTokens } from '../lib/theme';

const ML = 'ml-service';

type Async<T> = { status: 'idle' } | { status: 'loading' } | { status: 'ok'; data: T; ms: number } | { status: 'error'; error: ApiError };

function useAsync<T>() {
  const [s, setS] = useState<Async<T>>({ status: 'idle' });
  const run = useCallback(async (fn: () => Promise<T>) => {
    setS({ status: 'loading' });
    const t0 = performance.now();
    try {
      const data = await fn();
      setS({ status: 'ok', data, ms: performance.now() - t0 });
    } catch (e) {
      setS({ status: 'error', error: toApiError(e) });
    }
  }, []);
  const reset = useCallback(() => setS({ status: 'idle' }), []);
  return [s, run, reset] as const;
}

const LENGTHS = [256, 512, 1024, 2048, 4096];

function parseIq(text: string): IQ {
  let obj: unknown;
  try {
    obj = JSON.parse(text);
  } catch {
    throw new Error('not valid JSON');
  }
  const o = obj as { i?: unknown; q?: unknown };
  if (!Array.isArray(o.i) || !Array.isArray(o.q)) throw new Error('expected an object with numeric arrays "i" and "q"');
  const i = o.i.map(Number);
  const q = o.q.map(Number);
  if (i.some((v) => !Number.isFinite(v)) || q.some((v) => !Number.isFinite(v))) throw new Error('arrays must contain finite numbers');
  if (i.length !== q.length) throw new Error(`i and q lengths differ (${i.length} vs ${q.length})`);
  if (i.length < 64 || i.length > 65536) throw new Error('need between 64 and 65536 samples');
  return { i, q };
}

function fmtTick(v: number): string {
  if (Math.abs(v) < 5e-3) v = 0; // avoid a signed zero tick
  const a = Math.abs(v);
  if (a >= 1e6) return `${fmtNum(v / 1e6, 1)}M`;
  if (a >= 1e3) return `${fmtNum(v / 1e3, 0)}k`;
  if (a >= 10) return fmtNum(v, 0);
  return fmtNum(v, 2);
}

function PsdChart({ d }: { d: DetectResult }) {
  const themeV = useThemeVersion();
  const tk = useMemo(() => chartTokens(), [themeV]); // eslint-disable-line react-hooks/exhaustive-deps
  const data = useMemo(
    () =>
      d.freqs.map((f, i) => ({
        f,
        db: d.levels_db[i],
        det: d.mask[i] ? d.levels_db[i] : null,
      })),
    [d],
  );
  return (
    <div>
      <div style={{ height: 250 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke={tk.grid} vertical={false} />
            <XAxis
              dataKey="f"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={fmtTick}
              stroke={tk.axis}
              tick={{ fill: tk.muted, fontSize: 10 }}
              tickLine={false}
            />
            <YAxis
              width={44}
              tickFormatter={(v: number) => fmtNum(v, 0)}
              stroke={tk.axis}
              tick={{ fill: tk.muted, fontSize: 10 }}
              tickLine={false}
              unit=""
            />
            <Tooltip
              isAnimationActive={false}
              contentStyle={{ background: tk.surface, border: `1px solid ${tk.axis}`, fontSize: 11 }}
              labelStyle={{ color: tk.muted }}
              itemStyle={{ color: tk.ink }}
              labelFormatter={(v) => `f ${fmtTick(Number(v))}`}
              formatter={(v: number, name: string) => [`${fmtNum(v, 1)} dB`, name === 'det' ? 'CFAR detection' : 'PSD']}
            />
            <Line type="linear" dataKey="db" stroke={tk.s1} strokeWidth={1.5} dot={false} isAnimationActive={false} name="db" />
            <Scatter dataKey="det" fill={tk.s2} isAnimationActive={false} name="det" shape="circle" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <span>
          <i className="sw" style={{ background: 'var(--s1)', borderRadius: 0, height: 2, verticalAlign: 2 }} />
          PSD (dB)
        </span>
        <span>
          <i className="sw" style={{ background: 'var(--s2)' }} />
          CFAR detection ({d.n_detections} bin{d.n_detections === 1 ? '' : 's'})
        </span>
        <span className="muted">frequency axis as returned by /detect</span>
      </div>
    </div>
  );
}

function TopK({ r }: { r: ClassifyResult }) {
  const themeV = useThemeVersion();
  const tk = useMemo(() => chartTokens(), [themeV]); // eslint-disable-line react-hooks/exhaustive-deps
  const data = r.predictions.map((p) => ({ name: p.modulation, prob: p.prob }));
  return (
    <div>
      <div style={{ height: Math.max(120, 30 * data.length + 30) }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 8 }} barSize={12}>
            <CartesianGrid stroke={tk.grid} horizontal={false} />
            <XAxis type="number" domain={[0, 1]} tickFormatter={(v: number) => fmtPct(v)} stroke={tk.axis} tick={{ fill: tk.muted, fontSize: 10 }} tickLine={false} />
            <YAxis type="category" dataKey="name" width={64} stroke={tk.axis} tick={{ fill: tk.ink2, fontSize: 11 }} tickLine={false} />
            <Tooltip
              isAnimationActive={false}
              cursor={{ fill: tk.grid }}
              contentStyle={{ background: tk.surface, border: `1px solid ${tk.axis}`, fontSize: 11 }}
              itemStyle={{ color: tk.ink }}
              formatter={(v: number) => [fmtPct(v, 1), 'probability']}
            />
            <Bar dataKey="prob" fill={tk.s1} radius={[0, 4, 4, 0]} isAnimationActive={false} label={{ position: 'right', fill: tk.ink2, fontSize: 11, formatter: (v: number) => fmtPct(v, 1) }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="legend">
        <span className="muted">
          top-{r.predictions.length}, {r.n_samples} samples, inference {fmtMs(r.latency_ms)}
        </span>
      </div>
    </div>
  );
}

function Result<T>({ s, height, children }: { s: Async<T>; height: number; children: (d: T, ms: number) => JSX.Element }) {
  if (s.status === 'idle') return <EmptyState message="generate or paste a signal" height={height} />;
  if (s.status === 'loading') return <EmptyState loading service={ML} height={height} />;
  if (s.status === 'error') return <EmptyState service={ML} error={s.error} height={height} />;
  return children(s.data, s.ms);
}

export function SignalLab() {
  const mods = usePoll(ml.modulations, 30000);
  const [mode, setMode] = useState<'synth' | 'paste'>('synth');
  const [modulation, setModulation] = useState('');
  const [snr, setSnr] = useState(10);
  const [seed, setSeed] = useState(0);
  const [length, setLength] = useState(1024);
  const [pasted, setPasted] = useState('');
  const [pasteErr, setPasteErr] = useState<string | null>(null);
  const [iq, setIq] = useState<IQ | null>(null);
  const [label, setLabel] = useState('');
  const [synth, runSynth] = useAsync<IQ>();
  const [det, runDet] = useAsync<DetectResult>();
  const [spec, runSpec] = useAsync<SpectrogramResult>();
  const [cls, runCls] = useAsync<ClassifyResult>();

  const modList = mods.data ?? [];
  const effMod = modulation || modList[0] || '';

  const analyze = (sig: IQ) => {
    setIq(sig);
    void runDet(() => ml.detect(sig));
    void runSpec(() => ml.spectrogram(sig));
    void runCls(() => ml.classify(sig, 5));
  };

  const generate = () => {
    if (!effMod) return;
    void runSynth(async () => {
      const r = await ml.synth({ modulation: effMod, snr_db: snr, length, seed });
      const sig = { i: r.i, q: r.q };
      setLabel(`${r.modulation} at ${fmtNum(r.snr_db, 0)} dB SNR, ${sig.i.length} samples, seed ${seed}`);
      analyze(sig);
      return sig;
    });
  };

  const usePasted = () => {
    try {
      const sig = parseIq(pasted);
      setPasteErr(null);
      setLabel(`pasted signal, ${sig.i.length} samples`);
      analyze(sig);
    } catch (e) {
      setPasteErr(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="page">
      <div className="page-head">
        <h2>Signal lab</h2>
        <span className="muted">server-side synthesis, CFAR detection, STFT and classification via ml-service</span>
      </div>
      <div className="grid lab-grid">
        <Panel title="Input" service={ML} error={mods.error} updatedAt={mods.updatedAt}>
          <div className="seg" role="group" aria-label="Input mode">
            <button type="button" aria-pressed={mode === 'synth'} onClick={() => setMode('synth')}>
              Synthesise
            </button>
            <button type="button" aria-pressed={mode === 'paste'} onClick={() => setMode('paste')}>
              Paste I/Q
            </button>
          </div>
          {mode === 'synth' ? (
            <>
              <div className="field">
                <label htmlFor="mod">modulation</label>
                <select id="mod" value={effMod} onChange={(e) => setModulation(e.target.value)} disabled={!modList.length}>
                  {modList.length === 0 && <option value="">{mods.error ? 'ml-service unreachable' : 'loading classes'}</option>}
                  {modList.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="snr">
                  SNR <span className="val">{snr} dB</span>
                </label>
                <input id="snr" type="range" min={-20} max={30} step={1} value={snr} onChange={(e) => setSnr(Number(e.target.value))} />
              </div>
              <div className="field">
                <label htmlFor="len">length</label>
                <select id="len" value={length} onChange={(e) => setLength(Number(e.target.value))}>
                  {LENGTHS.map((n) => (
                    <option key={n} value={n}>
                      {n} samples
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="seed">seed</label>
                <input id="seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} />
              </div>
              <button className="btn primary" type="button" onClick={generate} disabled={!effMod || synth.status === 'loading'}>
                {synth.status === 'loading' ? 'Generating' : 'Generate and analyse'}
              </button>
              {synth.status === 'error' && (
                <p className="muted mono" style={{ color: 'var(--serious)', fontSize: 11 }}>
                  synth failed: {synth.error.message}
                </p>
              )}
            </>
          ) : (
            <>
              <div className="field">
                <label htmlFor="paste">JSON with i and q arrays (64 to 65536 samples)</label>
                <textarea
                  id="paste"
                  rows={10}
                  value={pasted}
                  onChange={(e) => setPasted(e.target.value)}
                  placeholder='{"i":[0.1,-0.3,...],"q":[0.2,0.4,...]}'
                  spellCheck={false}
                />
              </div>
              <button className="btn primary" type="button" onClick={usePasted} disabled={!pasted.trim()}>
                Analyse pasted signal
              </button>
              {pasteErr && (
                <p className="mono" style={{ color: 'var(--serious)', fontSize: 11 }} role="alert">
                  {pasteErr}
                </p>
              )}
            </>
          )}
          {iq && (
            <p className="muted mono" style={{ fontSize: 11, marginTop: 12 }}>
              current: {label}
            </p>
          )}
        </Panel>

        <div className="grid lab-results">
          <Panel title="Constellation" tools={iq ? <span>I vs Q</span> : undefined}>
            {iq ? <Constellation iq={iq} height={280} /> : <EmptyState message="generate or paste a signal" height={280} />}
          </Panel>
          <Panel title="Classifier top-5" tools={cls.status === 'ok' ? <span>round trip {fmtMs(cls.ms)}</span> : undefined}>
            <Result s={cls} height={280}>
              {(d) => <TopK r={d} />}
            </Result>
          </Panel>
          <Panel title="PSD with CFAR detections" tools={det.status === 'ok' ? <span>round trip {fmtMs(det.ms)}</span> : undefined}>
            <Result s={det} height={280}>
              {(d) => <PsdChart d={d} />}
            </Result>
          </Panel>
          <Panel title="Spectrogram" tools={spec.status === 'ok' ? <span>round trip {fmtMs(spec.ms)}</span> : undefined}>
            <Result s={spec} height={280}>
              {(d) => <SpectrogramCanvas data={d} height={280} />}
            </Result>
          </Panel>
        </div>
      </div>
    </div>
  );
}

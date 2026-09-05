import { useMemo } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { gateway, ml, type ApiError } from '../api';
import { chipLabel, useHealth, type ChipState } from '../hooks/useHealth';
import { EmptyState, Panel } from '../components/Panel';
import { StatTile } from '../components/StatTile';
import { usePoll } from '../hooks/usePoll';
import { useThemeVersion } from '../hooks/useTheme';
import { fmtInt, fmtMs, fmtNum, fmtUptime } from '../lib/format';
import { chartTokens } from '../lib/theme';

const ML = 'ml-service';

function Pill({ state }: { state: ChipState }) {
  return (
    <span className="state-pill" data-state={state}>
      <span className="dot" aria-hidden />
      {chipLabel(state)}
    </span>
  );
}

function CountBars({ title, counts, error, loading, service }: { title: string; counts: Record<string, number> | null; error: ApiError | null; loading: boolean; service: string }) {
  const themeV = useThemeVersion();
  const tk = useMemo(() => chartTokens(), [themeV]); // eslint-disable-line react-hooks/exhaustive-deps
  const data = useMemo(
    () =>
      Object.entries(counts ?? {})
        .map(([name, n]) => ({ name, n }))
        .sort((a, b) => b.n - a.n),
    [counts],
  );
  return (
    <Panel title={title} service={service} error={error} stale={!!error && data.length > 0} tools={<span>{data.length} categories</span>}>
      {data.length === 0 ? (
        <EmptyState service={service} error={error} loading={loading} message="no counts yet" height={160} />
      ) : (
        <div style={{ height: Math.max(140, 22 * data.length + 30) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 4, right: 56, bottom: 4, left: 8 }} barSize={10}>
              <CartesianGrid stroke={tk.grid} horizontal={false} />
              <XAxis type="number" tickFormatter={(v: number) => fmtInt(v)} stroke={tk.axis} tick={{ fill: tk.muted, fontSize: 10 }} tickLine={false} allowDecimals={false} />
              <YAxis type="category" dataKey="name" width={84} stroke={tk.axis} tick={{ fill: tk.ink2, fontSize: 11 }} tickLine={false} interval={0} />
              <Tooltip
                isAnimationActive={false}
                cursor={{ fill: tk.grid }}
                contentStyle={{ background: tk.surface, border: `1px solid ${tk.axis}`, fontSize: 11 }}
                itemStyle={{ color: tk.ink }}
                formatter={(v: number) => [fmtInt(v), 'detections']}
              />
              <Bar dataKey="n" fill={tk.s1} radius={[0, 4, 4, 0]} isAnimationActive={false} label={{ position: 'right', fill: tk.ink2, fontSize: 11, formatter: (v: number) => fmtInt(v) }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Panel>
  );
}

export function System() {
  const h = useHealth();
  const stats = usePoll(ml.stats, 3000);
  const ingest = usePoll(gateway.ingestStats, 5000);
  const s = stats.data;
  const mlh = h.ml.data;
  const agh = h.agent.data;
  const gwh = h.gateway.data;
  const lat = s?.latency_ms;
  const latMax = lat ? Math.max(lat.p99, 0.001) : 1;

  return (
    <div className="page">
      <div className="page-head">
        <h2>System</h2>
        <span className="muted">backend health, model, latency and throughput</span>
      </div>

      <div className="grid cols-4">
        <Panel title="ml-service" className="health-card" service={ML} error={h.ml.error} updatedAt={h.ml.updatedAt}>
          <div className="head">
            <Pill state={h.mlState} />
            <span className="muted mono">GET /api/ml/health</span>
          </div>
          {mlh ? (
            <dl className="kv">
              <dt>status</dt>
              <dd>{mlh.status}</dd>
              <dt>model loaded</dt>
              <dd>{mlh.model_loaded ? 'yes' : 'no'}</dd>
              <dt>version</dt>
              <dd>{mlh.version}</dd>
              <dt>uptime</dt>
              <dd>{fmtUptime(mlh.uptime_s)}</dd>
            </dl>
          ) : (
            <EmptyState service={ML} error={h.ml.error} loading={h.ml.loading} height={80} />
          )}
        </Panel>
        <Panel title="agent" className="health-card" service="agent" error={h.agent.error} updatedAt={h.agent.updatedAt}>
          <div className="head">
            <Pill state={h.agentState} />
            <span className="muted mono">GET /api/agent/health</span>
          </div>
          {agh ? (
            <dl className="kv">
              <dt>status</dt>
              <dd>{agh.status}</dd>
              <dt>LLM</dt>
              <dd>
                <Pill state={h.llmState} />
              </dd>
              <dt>model</dt>
              <dd>{agh.llm?.model ?? '-'}</dd>
              <dt>base URL</dt>
              <dd>{agh.llm?.base_url ?? '-'}</dd>
            </dl>
          ) : (
            <EmptyState service="agent" error={h.agent.error} loading={h.agent.loading} height={80} />
          )}
        </Panel>
        <Panel title="gateway" className="health-card" service="gateway" error={h.gateway.error} updatedAt={h.gateway.updatedAt}>
          <div className="head">
            <Pill state={h.gatewayState} />
            <span className="muted mono">GET /api/gateway/actuator/health</span>
          </div>
          {gwh ? (
            <dl className="kv">
              <dt>status</dt>
              <dd>{gwh.status}</dd>
              {Object.entries(gwh.components ?? {}).map(([k, v]) => (
                <span key={k} style={{ display: 'contents' }}>
                  <dt>{k}</dt>
                  <dd>{v.status}</dd>
                </span>
              ))}
            </dl>
          ) : (
            <EmptyState service="gateway" error={h.gateway.error} loading={h.gateway.loading} height={80} />
          )}
        </Panel>
        <Panel title="ingest" className="health-card" service="gateway" error={ingest.error} updatedAt={ingest.updatedAt} stale={!!ingest.error && !!ingest.data}>
          <div className="head">
            <span className="muted mono">GET /api/gateway/ingest/stats</span>
          </div>
          {ingest.data ? (
            <dl className="kv">
              <dt>accepted</dt>
              <dd>{fmtInt(ingest.data.accepted_total)}</dd>
              <dt>I/Q batches</dt>
              <dd>{fmtInt(ingest.data.iq_total)}</dd>
              <dt>PDW batches</dt>
              <dd>{fmtInt(ingest.data.pdw_total)}</dd>
              <dt>rejected</dt>
              <dd>{fmtInt(ingest.data.rejected_total)}</dd>
            </dl>
          ) : (
            <EmptyState service="gateway" error={ingest.error} loading={ingest.loading} height={80} />
          )}
        </Panel>
      </div>

      <div className="grid cols-3">
        <Panel title="Model" service={ML} error={h.ml.error}>
          {mlh ? (
            <dl className="kv">
              <dt>name</dt>
              <dd>{mlh.model}</dd>
              <dt>classes</dt>
              <dd>{fmtInt(mlh.classes)}</dd>
              <dt>providers</dt>
              <dd>{mlh.providers?.length ? mlh.providers.join(', ') : '-'}</dd>
              <dt>service version</dt>
              <dd>{mlh.version}</dd>
            </dl>
          ) : (
            <EmptyState service={ML} error={h.ml.error} loading={h.ml.loading} height={100} />
          )}
        </Panel>
        <Panel title="Detection latency" service={ML} error={stats.error} updatedAt={stats.updatedAt} stale={!!stats.error && !!s}>
          {lat ? (
            <div className="hbars" aria-label="Latency percentiles">
              {(['p50', 'p95', 'p99'] as const).map((k) => (
                <div className="hbar" key={k}>
                  <span className="k">{k}</span>
                  <span className="track">
                    <i style={{ width: `${Math.min(100, (lat[k] / latMax) * 100)}%` }} />
                  </span>
                  <span className="v">{fmtMs(lat[k])}</span>
                </div>
              ))}
              <span className="muted mono" style={{ fontSize: 11 }}>
                per-detection inference, {fmtInt(s?.detections_total)} detections total
              </span>
            </div>
          ) : (
            <EmptyState service={ML} error={stats.error} loading={stats.loading} height={100} />
          )}
        </Panel>
        <div className="grid cols-2" style={{ alignContent: 'start' }}>
          <StatTile label="Detections 10 min" value={s ? fmtInt(s.detections_10m) : null} stale={!!stats.error && !!s} />
          <StatTile label="Tracks" value={s ? fmtInt(s.tracks) : null} stale={!!stats.error && !!s} />
          <StatTile label="Low confidence 10 min" value={s ? fmtInt(s.low_confidence_10m) : null} stale={!!stats.error && !!s} />
          <StatTile label="p99 latency" value={lat ? fmtNum(lat.p99, 2) : null} unit={lat ? 'ms' : undefined} stale={!!stats.error && !!s} />
        </div>
      </div>

      <div className="grid cols-2">
        <CountBars title="Detections per modulation" counts={s?.per_modulation ?? null} error={stats.error} loading={stats.loading} service={ML} />
        <CountBars title="Detections per sensor" counts={s?.per_sensor ?? null} error={stats.error} loading={stats.loading} service={ML} />
      </div>
    </div>
  );
}

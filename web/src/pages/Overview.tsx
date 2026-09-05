import { useMemo, useState } from 'react';
import { ml } from '../api';
import { BearingRose } from '../components/BearingRose';
import { DetectionsTable } from '../components/DetectionsTable';
import { EmptyState, Panel } from '../components/Panel';
import { StatTile } from '../components/StatTile';
import { Waterfall } from '../components/Waterfall';
import { usePoll } from '../hooks/usePoll';
import { fmtInt, fmtMs, fmtNum, fmtPct, fmtTime } from '../lib/format';

const ML = 'ml-service';

export function Overview() {
  const stats = usePoll(ml.stats, 3000);
  const sensors = usePoll(ml.sensors, 10000);
  const [sensor, setSensor] = useState('');
  const hist = usePoll(() => ml.spectrumHistory({ sensor: sensor || undefined, n: 80 }), 1500, [sensor]);
  const dets = usePoll(() => ml.detections({ limit: 40 }), 1500);
  const eob = usePoll(ml.eob, 3000);

  const s = stats.data;
  const lowShare = s ? (s.detections_10m > 0 ? s.low_confidence_10m / s.detections_10m : 0) : null;
  const rows = hist.data?.rows ?? [];
  const centre = rows.length ? rows[rows.length - 1].rf_mhz : null;
  const tracks = useMemo(() => eob.data ?? [], [eob.data]);

  return (
    <div className="page">
      <div className="grid cols-4">
        <StatTile
          label="Detections, last 10 min"
          value={s ? fmtInt(s.detections_10m) : null}
          foot={s ? `${fmtInt(s.detections_total)} total` : stats.error ? `${ML} unreachable` : 'waiting'}
          stale={!!stats.error && !!s}
        />
        <StatTile
          label="Active tracks"
          value={s ? fmtInt(s.tracks) : null}
          foot={s ? 'fused EOB' : stats.error ? `${ML} unreachable` : 'waiting'}
          stale={!!stats.error && !!s}
        />
        <StatTile
          label="Low-confidence share"
          value={lowShare === null ? null : fmtPct(lowShare, 1)}
          foot={s ? `${fmtInt(s.low_confidence_10m)} below 0.60 in 10 min` : stats.error ? `${ML} unreachable` : 'waiting'}
          stale={!!stats.error && !!s}
        />
        <StatTile
          label="Detection latency p50"
          value={s ? fmtNum(s.latency_ms.p50, 2) : null}
          unit={s ? 'ms' : undefined}
          foot={s ? `p95 ${fmtMs(s.latency_ms.p95)}, p99 ${fmtMs(s.latency_ms.p99)}` : stats.error ? `${ML} unreachable` : 'waiting'}
          stale={!!stats.error && !!s}
        />
      </div>

      <div className="grid overview-mid">
        <Panel
          title="Spectrum waterfall"
          service={ML}
          error={hist.error}
          updatedAt={hist.updatedAt}
          stale={!!hist.error && rows.length > 0}
          tools={
            <>
              <label htmlFor="sensor-sel">sensor</label>
              <select
                id="sensor-sel"
                className="compact"
                value={sensor}
                onChange={(e) => setSensor(e.target.value)}
                disabled={!sensors.data?.length}
              >
                <option value="">latest of any</option>
                {(sensors.data ?? []).map((x) => (
                  <option key={x.sensor_id} value={x.sensor_id}>
                    {x.sensor_id} ({x.kind}, {x.detections_10m} / 10 min)
                  </option>
                ))}
              </select>
              {sensors.error && <span title={sensors.error.message}>sensor list unavailable</span>}
              {hist.data && <span>{hist.data.sensor_id || sensor || '-'}</span>}
              {rows.length > 0 && <span>{fmtTime(rows[rows.length - 1].ts)}Z</span>}
            </>
          }
        >
          {rows.length === 0 ? (
            <EmptyState
              service={ML}
              error={hist.error}
              loading={hist.loading}
              message="no spectrum sweeps received yet - start the sensor simulator"
              height={340 + 26}
            />
          ) : (
            <Waterfall rows={rows} centreMhz={centre} height={340} />
          )}
        </Panel>

        <Panel
          title="EOB bearing / frequency"
          service={ML}
          error={eob.error}
          updatedAt={eob.updatedAt}
          stale={!!eob.error && tracks.length > 0}
          tools={<span>{tracks.length} tracks</span>}
        >
          {tracks.length === 0 && (eob.error || eob.loading) ? (
            <EmptyState service={ML} error={eob.error} loading={eob.loading} height={340 + 26} />
          ) : (
            <BearingRose tracks={tracks} height={340} compact />
          )}
        </Panel>
      </div>

      <Panel
        title="Live detections"
        flush
        service={ML}
        error={dets.error}
        updatedAt={dets.updatedAt}
        stale={!!dets.error && !!dets.data?.length}
        tools={<span>newest first, limit 40, red tint below 0.60</span>}
      >
        {!dets.data?.length ? (
          <EmptyState
            service={ML}
            error={dets.error}
            loading={dets.loading}
            message="no detections yet - start the sensor simulator to build the picture"
            height={120}
          />
        ) : (
          <DetectionsTable rows={dets.data} maxHeight={380} />
        )}
      </Panel>
    </div>
  );
}

import type { Detection } from '../api';
import { fmtDeg, fmtMs, fmtNum, fmtTime } from '../lib/format';

export function SourceBadge({ source }: { source: string }) {
  return (
    <span className="badge" data-source={source}>
      {source}
    </span>
  );
}

export function ConfidenceBar({ value }: { value: number }) {
  const low = value < 0.6;
  return (
    <span className={`conf ${low ? 'low' : ''}`} title={low ? 'low confidence (below 0.60)' : undefined}>
      <span className="bar" aria-hidden>
        <i style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
      </span>
      <span className="v">{fmtNum(value, 2)}</span>
      {low && <span className="sr-only">low confidence</span>}
    </span>
  );
}

export function DetectionsTable({ rows, maxHeight = 360 }: { rows: Detection[]; maxHeight?: number }) {
  return (
    <div className="tbl-wrap" style={{ maxHeight }}>
      <table className="tbl">
        <thead>
          <tr>
            <th>time</th>
            <th>sensor</th>
            <th>source</th>
            <th className="num">RF MHz</th>
            <th className="num">AOA</th>
            <th>modulation</th>
            <th>confidence</th>
            <th className="num">latency</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d) => (
            <tr key={d.id} className={d.confidence < 0.6 ? 'low' : undefined}>
              <td className="num">{fmtTime(d.ts)}</td>
              <td className="mono">{d.sensor_id}</td>
              <td>
                <SourceBadge source={d.source} />
              </td>
              <td className="num">{fmtNum(d.rf_mhz, 1)}</td>
              <td className="num">{d.aoa === null ? '-' : fmtDeg(d.aoa)}</td>
              <td className="mono">{d.modulation}</td>
              <td>
                <ConfidenceBar value={d.confidence} />
              </td>
              <td className="num">{fmtMs(d.latency_ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

import { useMemo, useState, type KeyboardEvent } from 'react';
import { ml, type Track } from '../api';
import { BearingRose } from '../components/BearingRose';
import { SourceBadge } from '../components/DetectionsTable';
import { EmptyState, Panel } from '../components/Panel';
import { usePoll, useTimer } from '../hooks/usePoll';
import { fmtAge, fmtDateTime, fmtDeg, fmtInt, fmtNum, fmtPct } from '../lib/format';

const ML = 'ml-service';

function Detail({ t, now }: { t: Track; now: number }) {
  return (
    <dl className="kv">
      <dt>track</dt>
      <dd>T{t.track_id}</dd>
      <dt>RF</dt>
      <dd>
        {fmtNum(t.rf_mhz, 3)} MHz ({t.rf_rate >= 0 ? '+' : ''}
        {fmtNum(t.rf_rate, 3)} MHz/s)
      </dd>
      <dt>AOA</dt>
      <dd>
        {fmtDeg(t.aoa)} deg ({t.aoa_rate >= 0 ? '+' : ''}
        {fmtNum(t.aoa_rate, 2)} deg/s)
      </dd>
      <dt>sources</dt>
      <dd>
        {t.sources.map((s) => (
          <SourceBadge key={s} source={s} />
        ))}
      </dd>
      <dt>sensors</dt>
      <dd>{t.sensors.join(', ') || '-'}</dd>
      <dt>hits / misses</dt>
      <dd>
        {fmtInt(t.hits)} / {fmtInt(t.misses)}
      </dd>
      <dt>modulation</dt>
      <dd>{t.modulation || '-'}</dd>
      <dt>label agreement</dt>
      <dd>{fmtPct(t.label_agreement, 0)}</dd>
      <dt>pulse width</dt>
      <dd>{t.pw_us === null ? '-' : `${fmtNum(t.pw_us, 2)} us`}</dd>
      <dt>PRI</dt>
      <dd>
        {t.pri_us === null ? '-' : `${fmtNum(t.pri_us, 1)} us`}
        {t.pri_type ? ` (${t.pri_type})` : ''}
      </dd>
      <dt>first seen</dt>
      <dd>{fmtDateTime(t.first_seen)}</dd>
      <dt>last seen</dt>
      <dd>
        {fmtDateTime(t.last_seen)} ({fmtAge(t.last_seen, now)} ago)
      </dd>
      <dt>updated</dt>
      <dd>{fmtDateTime(t.updated)}</dd>
    </dl>
  );
}

export function Emitters() {
  const eob = usePoll(ml.eob, 2000);
  const [selected, setSelected] = useState<number | null>(null);
  const now = useTimer(1000).getTime();
  const tracks = useMemo(() => eob.data ?? [], [eob.data]);
  const sel = tracks.find((t) => t.track_id === selected) ?? null;

  const onRowKey = (e: KeyboardEvent<HTMLTableRowElement>, id: number) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setSelected(selected === id ? null : id);
    }
  };

  return (
    <div className="page">
      <div className="page-head">
        <h2>Electronic Order of Battle</h2>
        <span className="muted">fused tracks, strongest first, polled every 2 s</span>
      </div>
      <div className="grid emitters-grid">
        <Panel
          title="Tracks"
          flush
          service={ML}
          error={eob.error}
          updatedAt={eob.updatedAt}
          stale={!!eob.error && tracks.length > 0}
          tools={<span>{tracks.length} tracks - click a row to highlight</span>}
        >
          {tracks.length === 0 ? (
            <EmptyState
              service={ML}
              error={eob.error}
              loading={eob.loading}
              message="EOB is empty - no fused tracks yet"
              height={200}
            />
          ) : (
            <div className="tbl-wrap" style={{ maxHeight: 'calc(100vh - 200px)' }}>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>track</th>
                    <th className="num">RF MHz</th>
                    <th className="num">AOA</th>
                    <th>sources</th>
                    <th>sensors</th>
                    <th className="num">hits</th>
                    <th>modulation</th>
                    <th>PRI</th>
                    <th className="num">age</th>
                  </tr>
                </thead>
                <tbody>
                  {tracks.map((t) => (
                    <tr
                      key={t.track_id}
                      className={`selectable ${t.track_id === selected ? 'selected' : ''}`}
                      tabIndex={0}
                      aria-selected={t.track_id === selected}
                      onClick={() => setSelected(selected === t.track_id ? null : t.track_id)}
                      onKeyDown={(e) => onRowKey(e, t.track_id)}
                    >
                      <td className="mono">T{t.track_id}</td>
                      <td className="num">{fmtNum(t.rf_mhz, 1)}</td>
                      <td className="num">{fmtDeg(t.aoa)}</td>
                      <td>
                        {t.sources.map((s) => (
                          <SourceBadge key={s} source={s} />
                        ))}
                      </td>
                      <td className="mono wrap">{t.sensors.join(', ')}</td>
                      <td className="num">
                        {fmtInt(t.hits)}
                        {t.misses > 0 && <span className="muted"> / {fmtInt(t.misses)}</span>}
                      </td>
                      <td className="mono">{t.modulation}</td>
                      <td className="mono">
                        {t.pri_type ?? '-'}
                        {t.pri_us !== null && <span className="muted"> {fmtNum(t.pri_us, 0)} us</span>}
                      </td>
                      <td className="num">{fmtAge(t.last_seen, now)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <div className="grid" style={{ alignContent: 'start' }}>
          <Panel title="Bearing rose" tools={<span>AOA vs log RF</span>}>
            {tracks.length === 0 && (eob.error || eob.loading) ? (
              <EmptyState service={ML} error={eob.error} loading={eob.loading} height={360 + 26} />
            ) : (
              <BearingRose tracks={tracks} selectedId={selected} onSelect={setSelected} height={360} />
            )}
          </Panel>
          <Panel title="Track detail" tools={sel ? <span>T{sel.track_id}</span> : undefined}>
            {sel ? (
              <Detail t={sel} now={now} />
            ) : selected !== null ? (
              <EmptyState message={`track T${selected} is no longer in the EOB`} height={120} />
            ) : (
              <EmptyState message="select a track in the table or on the rose" height={120} />
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

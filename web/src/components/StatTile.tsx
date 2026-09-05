interface Props {
  label: string;
  value: string | null;
  unit?: string;
  foot?: string;
  stale?: boolean;
}

/** Stat tile. Proportional figures on the value (per dataviz guidance). */
export function StatTile({ label, value, unit, foot, stale }: Props) {
  return (
    <div className={`tile ${stale ? 'stale' : ''}`}>
      <div className="label">{label}</div>
      {value === null ? (
        <div className="value na" aria-label={`${label}: unavailable`}>
          n/a
        </div>
      ) : (
        <div className="value">
          {value}
          {unit && <span className="unit">{unit}</span>}
        </div>
      )}
      {foot && <div className="foot">{foot}</div>}
    </div>
  );
}

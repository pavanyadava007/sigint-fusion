export function fmtNum(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  return v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtInt(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  return Math.round(v).toLocaleString('en-US');
}

export function fmtMhz(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  return `${fmtNum(v, 1)} MHz`;
}

export function fmtFreqShort(mhz: number): string {
  if (mhz >= 1000) return `${fmtNum(mhz / 1000, mhz >= 10000 ? 1 : 2)} GHz`;
  return `${fmtNum(mhz, 0)} MHz`;
}

export function fmtDeg(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  return `${((v % 360) + 360) % 360 < 10 ? '00' : ((v % 360) + 360) % 360 < 100 ? '0' : ''}${fmtNum(((v % 360) + 360) % 360, 1)}`;
}

export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  return `${fmtNum(v * 100, digits)}%`;
}

/** HH:MM:SS UTC from an ISO timestamp. */
export function fmtTime(ts: string | null | undefined): string {
  if (!ts) return '-';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts.slice(11, 19) || ts;
  return d.toISOString().slice(11, 19);
}

export function fmtDateTime(ts: string | null | undefined): string {
  if (!ts) return '-';
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
}

/** Age in seconds relative to now, rendered as "12 s" / "3.5 min" / "2 h". */
export function fmtAge(ts: string | null | undefined, now: number = Date.now()): string {
  if (!ts) return '-';
  const t = new Date(ts).getTime();
  if (Number.isNaN(t)) return '-';
  const s = Math.max(0, (now - t) / 1000);
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${fmtNum(s / 60, 1)} min`;
  if (s < 86400) return `${fmtNum(s / 3600, 1)} h`;
  return `${fmtNum(s / 86400, 1)} d`;
}

export function fmtUptime(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return '-';
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m ${Math.floor(s % 60)}s`;
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '-';
  if (v >= 1000) return `${fmtNum(v / 1000, 2)} s`;
  return `${fmtNum(v, v < 10 ? 2 : 1)} ms`;
}

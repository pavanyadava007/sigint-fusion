// Analyst for browser mode. Evidence gathering runs in the tab (frequency extraction + emitter
// catalogue lookup, lexical retrieval over the knowledge base chunks, live picture summary, EOB
// table); the evidence is sent as compact JSON context to Qwen2.5-7B on a ZeroGPU Space
// (src/browser/llm.ts). When that backend is asleep, over quota, disabled or slower than 120 s the
// same evidence is answered by the rule-based templates below. Shape matches POST /api/agent/ask.

import type { CatalogueEmitter, CorpusChunk } from './bank';
import { round } from './dsp';
import type { EobTrack } from './fusion';
import { askLlm, BACKEND, shortReason } from './llm';
import type { DetectionRow, ReportRow, Store } from './sim';

export const ANALYST_MODEL = 'rule-based analyst';
export const FALLBACK_MODEL = 'rule-based analyst (fallback)';
const CONTEXT_MAX_CHARS = 8000;

export interface TraceStep {
  tool: string;
  args: Record<string, unknown>;
  result_preview: string;
  seconds: number;
}

export interface AskResult {
  answer: string;
  trace: TraceStep[];
  seconds: number;
  model: string;
}

// ---------- frequency extraction + catalogue ----------

const FREQ_RE = /(\d+(?:[.,]\d+)?)\s*(ghz|mhz|khz)\b/gi;

export function extractFreqsMhz(q: string): number[] {
  const out: number[] = [];
  for (const m of q.matchAll(FREQ_RE)) {
    const v = Number(m[1].replace(',', '.'));
    const unit = m[2].toLowerCase();
    const mhz = unit === 'ghz' ? v * 1000 : unit === 'khz' ? v / 1000 : v;
    if (Number.isFinite(mhz) && !out.includes(mhz)) out.push(mhz);
  }
  return out;
}

export function toleranceMhz(f: number): number {
  return Math.max(2, 0.02 * f);
}

export function lookupEmitters(cat: CatalogueEmitter[], f: number, tol = toleranceMhz(f)): CatalogueEmitter[] {
  return cat.filter((e) => e.rf_min_mhz - tol <= f && e.rf_max_mhz + tol >= f).sort((a, b) => a.rf_max_mhz - a.rf_min_mhz - (b.rf_max_mhz - b.rf_min_mhz));
}

// ---------- lexical retrieval ----------

const STOP = new Set(
  'a an and are as at be by for from has have how in is it its of on or that the this to was what when where which who why with near around about any should there'.split(' '),
);

function tokens(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9.]+/)
    .map((t) => t.replace(/^\.+|\.+$/g, ''))
    .filter((t) => t.length >= 2 && !STOP.has(t));
}

export interface Hit extends CorpusChunk {
  score: number;
}

export class Index {
  private readonly docs: { chunk: CorpusChunk; tf: Map<string, number>; len: number }[];
  private readonly df = new Map<string, number>();
  private readonly avgLen: number;
  constructor(chunks: CorpusChunk[]) {
    this.docs = chunks.map((chunk) => {
      const tf = new Map<string, number>();
      const toks = tokens(chunk.text);
      for (const t of toks) tf.set(t, (tf.get(t) ?? 0) + 1);
      for (const t of tf.keys()) this.df.set(t, (this.df.get(t) ?? 0) + 1);
      return { chunk, tf, len: toks.length };
    });
    this.avgLen = this.docs.reduce((a, d) => a + d.len, 0) / Math.max(1, this.docs.length);
  }
  /** BM25-style scoring (k1 1.2, b 0.75) over the query terms. */
  search(query: string, k = 5): Hit[] {
    const qt = Array.from(new Set(tokens(query)));
    const N = this.docs.length;
    const hits: Hit[] = [];
    for (const d of this.docs) {
      let score = 0;
      for (const t of qt) {
        const tf = d.tf.get(t);
        if (!tf) continue;
        const df = this.df.get(t) ?? 0;
        const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
        score += idf * ((tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (d.len / this.avgLen))));
      }
      if (score > 0) hits.push({ ...d.chunk, score });
    }
    return hits.sort((a, b) => b.score - a.score).slice(0, k);
  }
}

// ---------- formatting helpers ----------

const f1 = (v: number | null | undefined, d = 1) => (v === null || v === undefined || !Number.isFinite(v) ? '-' : round(v, d).toFixed(d));

function band(e: CatalogueEmitter): string {
  return e.rf_min_mhz === e.rf_max_mhz ? `${f1(e.rf_min_mhz)} MHz` : `${f1(e.rf_min_mhz)}-${f1(e.rf_max_mhz)} MHz`;
}

function catLine(e: CatalogueEmitter): string {
  const parts = [`${band(e)}`, e.modulation];
  if (e.pri_us) parts.push(`PRI ${f1(e.pri_us, 0)} us`);
  if (e.pw_us) parts.push(`PW ${f1(e.pw_us, 2)} us`);
  return `**${e.name}** (${e.type}): ${parts.join(', ')}. ${e.notes}`;
}

function trackLine(t: EobTrack): string {
  const pri = t.pri_type ? `, PRI ${t.pri_type}${t.pri_us !== null ? ` ${f1(t.pri_us, 0)} us` : ''}` : '';
  const pw = t.pw_us !== null ? `, PW ${f1(t.pw_us, 2)} us` : '';
  return `T${t.track_id} at ${f1(t.rf_mhz)} MHz, AOA ${f1(t.aoa)} deg, ${t.sources.join('+')} via ${t.sensors.join(', ')}, ${t.modulation ?? '-'}${pri}${pw}, ${t.hits} hits`;
}

function bestMatch(cat: CatalogueEmitter[], t: EobTrack): CatalogueEmitter | null {
  const c = lookupEmitters(cat, t.rf_mhz);
  if (!c.length) return null;
  const isPulse = t.modulation === 'pulse' || t.sources.includes('resm');
  const same = c.filter((e) => (e.modulation === 'pulse') === isPulse && (!isPulse || !t.modulation || e.modulation === 'pulse') && (isPulse || !t.modulation || e.modulation === t.modulation));
  return (same.length ? same : c)[0];
}

function eobTable(tracks: EobTrack[], cat: CatalogueEmitter[]): string {
  const head = '| track | RF MHz | AOA | sources | sensors | modulation | PRI | PW us | hits | catalogue match |\n|---|---|---|---|---|---|---|---|---|---|';
  const rows = tracks.map((t) => {
    const m = bestMatch(cat, t);
    const pri = t.pri_type ? `${t.pri_type}${t.pri_us !== null ? ` ${f1(t.pri_us, 0)} us` : ''}` : '-';
    return `| T${t.track_id} | ${f1(t.rf_mhz)} | ${f1(t.aoa)} | ${t.sources.join('+')} | ${t.sensors.join(', ')} | ${t.modulation ?? '-'} | ${pri} | ${t.pw_us === null ? '-' : f1(t.pw_us, 2)} | ${t.hits} | ${m ? m.name : 'none'} |`;
  });
  return [head, ...rows].join('\n');
}

function counts<T>(rows: T[], key: (r: T) => string): [string, number][] {
  const m = new Map<string, number>();
  for (const r of rows) m.set(key(r), (m.get(key(r)) ?? 0) + 1);
  return Array.from(m.entries()).sort((a, b) => b[1] - a[1]);
}

// ---------- the analyst ----------

const REPORT_RE = /\b(write|produce|generate|create|make|draft|prepare|compile|save)\b[^.?!]*\breport\b|\breport on\b/i;
const LIST_REPORTS_RE = /\b(list|show|open|which|what|any)\b[^.?!]*\breports?\b|\breports? (list|so far|saved)\b/i;
const PICTURE_RE = /\b(detection|detections|picture|transmitting|transmit|order of battle|eob|tracks?|activity|emitters?|sensors?|summari[sz]e|summary|last \d+ min|low[- ]?confidence|confidence|re-?check|what is (it|that)|seen)\b/i;
const MULTI_SENSOR_RE = /more than one sensor|multiple sensors|several sensors|both sensors|two sensors|fused|multi-?sensor/i;
const LOW_CONF_RE = /low[- ]?confidence|re-?check|uncertain|tentative/i;

interface FreqLookup {
  f: number;
  tol: number;
  matches: CatalogueEmitter[];
  near: EobTrack[];
}

interface Evidence {
  q: string;
  cat: CatalogueEmitter[];
  freqs: number[];
  lookups: FreqLookup[];
  wantsReport: boolean;
  wantsList: boolean;
  wantsPicture: boolean;
  tracks: EobTrack[];
  recent: DetectionRow[];
  reports: ReportRow[];
  hits: Hit[];
  trace: TraceStep[];
}

export class Analyst {
  private readonly index: Index;
  constructor(private readonly store: Store) {
    this.index = new Index(store.assets.corpus.chunks);
  }

  /** Run the in-tab tools and record them in the trace. */
  private gather(question: string): Evidence {
    const trace: TraceStep[] = [];
    const timed = <T>(tool: string, args: Record<string, unknown>, fn: () => T, preview: (r: T) => string): T => {
      const t0 = performance.now();
      const r = fn();
      trace.push({ tool, args, result_preview: preview(r).slice(0, 300), seconds: round((performance.now() - t0) / 1000, 4) });
      return r;
    };
    const cat = this.store.assets.corpus.emitters;
    const q = question.trim();
    const freqs = extractFreqsMhz(q);
    const wantsReport = REPORT_RE.test(q);
    const wantsList = !wantsReport && LIST_REPORTS_RE.test(q);
    const wantsPicture = PICTURE_RE.test(q) || wantsReport;
    const tracks = timed('current_tracks', {}, () => this.store.tracks, (r) => `${r.length} tracks: ${r.map((t) => `T${t.track_id} ${f1(t.rf_mhz)} MHz`).join(', ')}`);
    const recent = timed('recent_detections', { minutes: 10, limit: 500 }, () => this.store.recentDetections(10, 500), (r) => `${r.length} detections in 10 min; per modulation ${counts(r, (d) => d.modulation).map(([k, n]) => `${k}=${n}`).join(', ')}`);
    const lookups: FreqLookup[] = freqs.map((f) => {
      const tol = toleranceMhz(f);
      const matches = timed('query_emitter_db', { rf_mhz: f, tolerance_mhz: round(tol, 1) }, () => lookupEmitters(cat, f, tol), (r) => (r.length ? r.map((e) => e.name).join('; ') : 'no catalogue match'));
      return { f, tol, matches, near: tracks.filter((t) => Math.abs(t.rf_mhz - f) <= tol) };
    });
    const reports = wantsList ? timed('list_reports', { limit: 20 }, () => this.store.reports.slice(-20).reverse(), (r) => `${r.length} reports`) : [];
    const hits = timed('search_docs', { query: q, k: 5 }, () => this.index.search(q, 5), (r) => (r.length ? r.map((h) => `[${h.source}] ${h.score.toFixed(2)}`).join(', ') : 'no documents found'));
    return { q, cat, freqs, lookups, wantsReport, wantsList, wantsPicture, tracks, recent, reports, hits, trace };
  }

  /** Compact JSON context for the LLM backend, capped at about 8 kB. */
  private buildContext(ev: Evidence): string {
    const trackRow = (t: EobTrack) => {
      const m = bestMatch(ev.cat, t);
      return { track_id: t.track_id, rf_mhz: t.rf_mhz, aoa: t.aoa, sources: t.sources, sensors: t.sensors, hits: t.hits, modulation: t.modulation, label_agreement: t.label_agreement, pw_us: t.pw_us, pri_us: t.pri_us === null ? null : round(t.pri_us, 1), pri_type: t.pri_type, catalogue_match: m ? m.name : null };
    };
    const low = ev.recent.filter((d) => d.confidence < 0.6);
    const catalogue = ev.lookups.map((l) => ({
      rf_mhz: l.f,
      tolerance_mhz: round(l.tol, 1),
      matches: l.matches.map((e) => ({ name: e.name, type: e.type, rf_min_mhz: e.rf_min_mhz, rf_max_mhz: e.rf_max_mhz, pri_us: e.pri_us ?? null, pw_us: e.pw_us ?? null, modulation: e.modulation, notes: e.notes })),
      live_tracks_near: l.near.map((t) => `T${t.track_id}`),
    }));
    const base = {
      catalogue,
      detections_summary: {
        minutes: 10,
        total: ev.recent.length,
        per_modulation: Object.fromEntries(counts(ev.recent, (d) => d.modulation)),
        per_sensor: Object.fromEntries(counts(ev.recent, (d) => d.sensor_id).sort((a, b) => a[0].localeCompare(b[0]))),
        low_confidence: low.length,
        low_confidence_examples: low.slice(0, 5).map((d) => `${d.sensor_id} ${f1(d.rf_mhz)} MHz ${d.modulation} ${d.confidence.toFixed(2)}`),
      },
      tracks: ev.tracks.map(trackRow),
      ...(ev.wantsList ? { reports: ev.reports.map((r) => ({ id: r.id, ts: r.ts, question: r.question })) } : {}),
    };
    let maxPassage = 500;
    let passages = ev.hits.map((h) => `[${h.source}] ${h.text.length > maxPassage ? `${h.text.slice(0, maxPassage).trimEnd()} ...` : h.text}`);
    let ctx = JSON.stringify({ ...base, passages });
    while (ctx.length > CONTEXT_MAX_CHARS && passages.length) {
      if (maxPassage > 200) maxPassage -= 100;
      else passages = passages.slice(0, -1);
      passages = passages.map((p) => (p.length > maxPassage + 40 ? `${p.slice(0, maxPassage + 30).trimEnd()} ...` : p));
      ctx = JSON.stringify({ ...base, passages });
    }
    return ctx;
  }

  /** Templated answer from the evidence (the pre-LLM behaviour). Saves a report when one was requested. */
  private ruleAnswer(ev: Evidence, thread: string, trace: TraceStep[]): string {
    const { q, cat, tracks, recent } = ev;
    const parts: string[] = [];
    const timed = <T>(tool: string, args: Record<string, unknown>, fn: () => T, preview: (r: T) => string): T => {
      const t0 = performance.now();
      const r = fn();
      trace.push({ tool, args, result_preview: preview(r).slice(0, 300), seconds: round((performance.now() - t0) / 1000, 4) });
      return r;
    };

    // (a) frequencies -> catalogue
    for (const { f, tol, matches, near } of ev.lookups) {
      const lines: string[] = [];
      if (near.length) {
        const t = near[0];
        const m = bestMatch(cat, t);
        lines.push(
          m
            ? `**Identification at ${f1(f)} MHz: ${m.name}.** The live EOB holds ${trackLine(t)}; RF sits inside the catalogue band ${band(m)}${m.pri_us && t.pri_us !== null ? `, PRI ${f1(t.pri_us, 0)} us vs catalogue ${f1(m.pri_us, 0)} us` : ''}${m.pw_us && t.pw_us !== null ? `, PW ${f1(t.pw_us, 2)} us vs catalogue ${f1(m.pw_us, 2)} us` : ''}.`
            : `**Live track near ${f1(f)} MHz with no catalogue entry:** ${trackLine(t)}.`,
        );
        for (const extra of near.slice(1)) lines.push(`Also near this frequency: ${trackLine(extra)}.`);
      } else {
        lines.push(`**No live track within ${f1(tol, 0)} MHz of ${f1(f)} MHz** in the current EOB (${tracks.length} tracks).`);
      }
      if (matches.length) {
        lines.push(`Catalogue entries whose band contains ${f1(f)} MHz (tolerance ${f1(tol, 0)} MHz):`);
        for (const e of matches) lines.push(`- ${catLine(e)}`);
      } else {
        lines.push(`No catalogue entry covers ${f1(f)} MHz within ${f1(tol, 0)} MHz.`);
      }
      parts.push(lines.join('\n'));
    }

    // (c) live picture
    if (ev.wantsPicture && !ev.wantsReport) {
      const perMod = counts(recent, (d) => d.modulation);
      const perSensor = counts(recent, (d) => d.sensor_id).sort((a, b) => a[0].localeCompare(b[0]));
      const low = recent.filter((d) => d.confidence < 0.6);
      const lines: string[] = [];
      lines.push(`**Live picture (last 10 min):** ${recent.length} detections from ${perSensor.length} sensors, ${tracks.length} active tracks, ${low.length} below 0.60 confidence.`);
      if (perMod.length) lines.push(`Per modulation: ${perMod.map(([k, n]) => `${k} ${n}`).join(', ')}.`);
      if (perSensor.length) lines.push(`Per sensor: ${perSensor.map(([k, n]) => `${k} ${n}`).join(', ')}.`);
      if (MULTI_SENSOR_RE.test(q)) {
        const multi = tracks.filter((t) => t.sensors.length > 1);
        lines.push(
          multi.length
            ? `Tracks seen by more than one sensor: ${multi.map((t) => `T${t.track_id} (${f1(t.rf_mhz)} MHz, ${t.sensors.join(' + ')}, sources ${t.sources.join('+')})`).join('; ')}. A track that combines comint and resm observations is the fused emitter.`
            : 'No track is currently seen by more than one sensor.',
        );
      }
      if (LOW_CONF_RE.test(q)) {
        lines.push(
          low.length
            ? `Low-confidence detections to re-check (${low.length}): ${low
                .slice(0, 8)
                .map((d) => `${d.sensor_id} ${f1(d.rf_mhz)} MHz ${d.modulation} ${d.confidence.toFixed(2)}`)
                .join('; ')}${low.length > 8 ? '; ...' : ''}. Mark these identifications tentative and request a longer capture on the same bearing.`
            : 'No detections below 0.60 confidence in the last 10 minutes.',
        );
      }
      if (tracks.length) {
        lines.push('');
        lines.push(eobTable(tracks, cat));
      }
      parts.push(lines.join('\n'));
    }

    // (d) report
    if (ev.wantsReport) {
      const report = this.composeReport(q, tracks, recent, cat);
      const saved = timed('save_report', { question: q, chars: report.length }, () => this.store.addReport(q, report, thread), (r) => `saved report #${r.id}`);
      parts.push(`Report saved as #${saved.id} (${new Date(saved.ts).toISOString().slice(0, 19)}Z); it is listed in the Reports panel.\n\n${report}`);
    }

    // (e) list reports
    if (ev.wantsList) {
      parts.push(
        ev.reports.length
          ? `Saved reports (${ev.reports.length}):\n${ev.reports.map((r) => `- #${r.id} ${new Date(r.ts).toISOString().slice(0, 19)}Z: ${r.question}`).join('\n')}`
          : 'No reports have been saved yet. Ask me to write a report on the current EOB to create one.',
      );
    }

    // (b) knowledge base
    if (ev.hits.length) {
      const lines = ['**Knowledge base:**'];
      for (const h of ev.hits) lines.push(`- [${h.source}] ${h.text.length > 220 ? `${h.text.slice(0, 220).trimEnd()} ...` : h.text}`);
      parts.push(lines.join('\n'));
    }

    if (!parts.length) {
      parts.push(
        'I could not match that question to the emitter catalogue, the live picture or the knowledge base. Try a frequency ("what is transmitting near 9400 MHz"), ask for a summary of the last 10 minutes, or ask me to write a report on the current EOB.',
      );
    }
    parts.push('_Rule-based answer (retrieval + templates) from the evidence gathered in this tab._');
    return parts.join('\n\n');
  }

  /** LLM first (Qwen2.5-7B on the ZeroGPU Space), rule-based templates on any failure. */
  async ask(question: string, thread = 'ui'): Promise<AskResult> {
    const tStart = performance.now();
    const ev = this.gather(question);
    const trace = ev.trace;
    const context = this.buildContext(ev);
    try {
      const t0 = performance.now();
      const llm = await askLlm(ev.q, context, ev.wantsReport);
      let answer = llm.answer.trim();
      if (ev.wantsReport) {
        const saved = this.store.addReport(ev.q, answer, thread);
        trace.push({ tool: 'save_report', args: { question: ev.q, chars: answer.length }, result_preview: `saved report #${saved.id}`, seconds: 0 });
        answer += `\n\nReport saved as #${saved.id}.`;
      }
      trace.push({
        tool: 'llm',
        args: { model: llm.model, backend: BACKEND, report: ev.wantsReport, context_chars: context.length, ...(llm.zerogpu !== undefined ? { zerogpu: llm.zerogpu } : {}) },
        result_preview: llm.answer.slice(0, 200),
        seconds: round((performance.now() - t0) / 1000, 2),
      });
      return { answer, trace, seconds: round((performance.now() - tStart) / 1000, 3), model: llm.model };
    } catch (e) {
      const reason = shortReason(e);
      const body = this.ruleAnswer(ev, thread, trace);
      const answer = `_LLM backend unavailable (${reason}), rule-based answer from the same evidence._\n\n${body}`;
      return { answer, trace, seconds: round((performance.now() - tStart) / 1000, 3), model: FALLBACK_MODEL };
    }
  }

  composeReport(question: string, tracks: EobTrack[], recent: DetectionRow[], cat: CatalogueEmitter[]): string {
    const now = new Date().toISOString().slice(0, 19) + 'Z';
    const fused = tracks.filter((t) => t.sources.length > 1);
    const low = recent.filter((d) => d.confidence < 0.6);
    const sensors = counts(recent, (d) => d.sensor_id).sort((a, b) => a[0].localeCompare(b[0]));
    const perMod = counts(recent, (d) => d.modulation);
    const lines: string[] = [];
    lines.push(`# Intelligence report, ${now}`);
    lines.push('');
    lines.push('## Summary');
    lines.push(
      `${tracks.length} active emitter tracks in the Electronic Order of Battle, built from ${recent.length} detections by ${sensors.length} sensors (${sensors.map(([k, n]) => `${k}: ${n}`).join(', ')}) in the last 10 minutes. ` +
        `${fused.length} track${fused.length === 1 ? '' : 's'} ${fused.length === 1 ? 'is' : 'are'} confirmed by more than one source${fused.length ? ` (${fused.map((t) => `T${t.track_id} at ${f1(t.rf_mhz)} MHz`).join(', ')})` : ''}. ` +
        `Modulations observed: ${perMod.map(([k, n]) => `${k} (${n})`).join(', ') || 'none'}.`,
    );
    lines.push('');
    lines.push('## Emitters');
    lines.push(tracks.length ? eobTable(tracks, cat) : 'No tracks in the EOB.');
    lines.push('');
    lines.push('## Assessment');
    for (const t of [...tracks].sort((a, b) => a.rf_mhz - b.rf_mhz)) {
      const m = bestMatch(cat, t);
      lines.push(
        m
          ? `- T${t.track_id} (${f1(t.rf_mhz)} MHz, AOA ${f1(t.aoa)} deg): assessed as ${m.name} (${band(m)}, ${m.modulation}${m.pri_us ? `, PRI ${f1(m.pri_us, 0)} us` : ''}). Observed ${t.modulation ?? '-'}${t.pri_type ? `, PRI ${t.pri_type}${t.pri_us !== null ? ` ${f1(t.pri_us, 0)} us` : ''}` : ''}${t.pw_us !== null ? `, PW ${f1(t.pw_us, 2)} us` : ''} by ${t.sensors.join(' and ')}.`
          : `- T${t.track_id} (${f1(t.rf_mhz)} MHz, AOA ${f1(t.aoa)} deg): no catalogue entry; observed ${t.modulation ?? '-'} by ${t.sensors.join(' and ')}.`,
      );
    }
    if (!tracks.length) lines.push('- Nothing to assess yet.');
    lines.push('');
    lines.push('## Confidence');
    lines.push(
      `- Multi-source tracks (comint + resm): high confidence, ${fused.length} track${fused.length === 1 ? '' : 's'}.\n` +
        `- Single-source tracks: medium confidence; label agreement ${tracks.length ? `${f1(Math.min(...tracks.map((t) => t.label_agreement ?? 0)) * 100, 0)}% to ${f1(Math.max(...tracks.map((t) => t.label_agreement ?? 0)) * 100, 0)}%` : '-'}.\n` +
        `- ${low.length} of ${recent.length} detections in the window fell below 0.60 classifier confidence${recent.length ? ` (${f1((100 * low.length) / recent.length, 1)}%)` : ''}.\n` +
        `- Classifier: ${this.store.assets.card.model}, validation top-1 ${f1(this.store.assets.card.val_top1 * 100, 1)}% on RadioML 2018.01A, running in the browser on real frames.`,
    );
    lines.push('');
    lines.push('## Recommended actions');
    lines.push('- Task a second COMINT sensor onto single-source comms tracks to obtain cross-bearings.');
    lines.push('- Re-capture any emitter whose label agreement is below 80% with a longer dwell before naming it.');
    if (fused.length) lines.push(`- Treat ${fused.map((t) => `T${t.track_id}`).join(', ')} as confirmed; compare PRI type and pulse width with the catalogue before final identification.`);
    lines.push('- Continue monitoring for bearing drift; the fuser reports rates in the track detail.');
    lines.push('');
    lines.push(`_Requested by: "${question}"_`);
    return lines.join('\n');
  }
}

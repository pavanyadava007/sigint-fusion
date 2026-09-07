// Browser-mode port of ml-service/sensor_sim.py + consumer.py: the fixed emitter scenario, three
// COMINT sensors classifying real RadioML frames with the ONNX model, one R-ESM sensor
// deinterleaving PDW batches, everything fused into the EOB. Keeps the in-memory tables that the
// fetch shim serves (detections, spectra, tracks, reports, gateway counters).

import type { Assets } from './bank';
import { deinterleavePdw, round, welchPsd } from './dsp';
import { Fuser, type EobTrack, type FusionDetection } from './fusion';
import { Classifier } from './model';
import { Rng } from './rng';

export interface DetectionRow {
  id: number;
  ts: string;
  t: number; // epoch ms, for range queries
  sensor_id: string;
  source: 'comint' | 'resm';
  rf_mhz: number;
  aoa: number | null;
  modulation: string;
  confidence: number;
  latency_ms: number;
  pw_us: number | null;
  pri_us: number | null;
  pri_type: string | null;
}

export interface SpectrumRow {
  ts: string;
  t: number;
  rf_mhz: number;
  levels: number[];
}

export interface ReportRow {
  id: number;
  ts: string;
  question: string;
  report: string;
  thread: string;
}

export interface Counters {
  accepted_total: number;
  iq_total: number;
  pdw_total: number;
  rejected_total: number;
}

interface Emitter {
  name: string;
  rf: number;
  aoa: number;
  mod: string;
  snr: number;
  radar: [number, number, 'constant' | 'stagger' | 'jitter'] | null; // pw us, pri ms, pri type
}

// rf MHz, bearing deg, modulation, snr dB, radar (pw us, pri ms, pri type) or null
export const SCENARIO: Emitter[] = [
  { name: 'marine VHF ch16', rf: 156.8, aoa: 42.0, mod: 'FM', snr: 18, radar: null },
  { name: 'airband tower', rf: 121.5, aoa: 305.0, mod: 'AM-DSB-WC', snr: 14, radar: null },
  { name: 'SSR reply', rf: 1090.0, aoa: 200.0, mod: 'OOK', snr: 12, radar: [0.5, 1.0, 'jitter'] }, // COMINT (OOK) and R-ESM (pulses) -> fused track
  { name: 'data link', rf: 2400.0, aoa: 95.0, mod: 'QPSK', snr: 8, radar: null },
  { name: 'S-band surveillance radar', rf: 2800.0, aoa: 270.0, mod: 'pulse', snr: 20, radar: [1.0, 1.0, 'constant'] },
  { name: 'X-band nav radar', rf: 9410.0, aoa: 118.0, mod: 'pulse', snr: 20, radar: [0.5, 0.5, 'stagger'] },
  { name: 'weather radar', rf: 5600.0, aoa: 15.0, mod: 'pulse', snr: 20, radar: [2.0, 1.2, 'jitter'] },
];

export const N_COMINT_SENSORS = 3;
export const ESM_SENSOR = 'sirius-esm';
export const TICK_MS = 250;
const DETECTIONS_CAP = 5000;
const SPECTRA_CAP = 120;
const FRAME_NOISE_SD = 0.05; // added to the RMS-normalised frame so repeated draws differ

/** 40 ms batch of pulse descriptor words [toa_ms, rf_mhz, pw_us, aoa_deg] from the radar emitters. */
export function pdwBatch(rng: Rng, t0Ms: number, durationMs = 40): number[][] {
  const rows: number[][] = [];
  for (const e of SCENARIO) {
    if (!e.radar) continue;
    const [pw, pri, kind] = e.radar;
    let t = t0Ms + rng.uniform(0, pri);
    let k = 0;
    while (t < t0Ms + durationMs) {
      const step = kind === 'constant' ? pri : kind === 'stagger' ? pri * (1 + 0.3 * (k % 3)) : pri * rng.uniform(0.85, 1.15);
      rows.push([t, e.rf + rng.normal(0, 0.5), pw * rng.uniform(0.98, 1.02), (((e.aoa + rng.normal(0, 1.5)) % 360) + 360) % 360]);
      t += step;
      k += 1;
    }
  }
  rows.sort((a, b) => a[0] - b[0]);
  return rows;
}

export class Store {
  readonly t0 = Date.now();
  readonly detections: DetectionRow[] = []; // oldest first
  readonly spectra = new Map<string, SpectrumRow[]>(); // per sensor, oldest first
  tracks: EobTrack[] = [];
  readonly reports: ReportRow[] = []; // oldest first
  readonly counters: Counters = { accepted_total: 0, iq_total: 0, pdw_total: 0, rejected_total: 0 };
  readonly fuser = new Fuser(4.0, 20);
  readonly rng = new Rng(7);
  tick = 0;
  private nextDetId = 1;
  private nextReportId = 1;
  private timer: number | undefined;
  private busy = false;
  private lastSpectrumTs = 0;

  constructor(
    readonly assets: Assets,
    readonly clf: Classifier,
  ) {}

  get uptimeS(): number {
    return round((Date.now() - this.t0) / 1000, 1);
  }

  start(): void {
    if (this.timer !== undefined) return;
    this.timer = window.setInterval(() => void this.step(), TICK_MS);
    void this.step();
  }

  stop(): void {
    if (this.timer !== undefined) window.clearInterval(this.timer);
    this.timer = undefined;
  }

  private pushDetection(d: Omit<DetectionRow, 'id' | 'ts' | 't'>, now: number): DetectionRow {
    const row: DetectionRow = { id: this.nextDetId++, ts: new Date(now).toISOString(), t: now, ...d };
    this.detections.push(row);
    if (this.detections.length > DETECTIONS_CAP) this.detections.splice(0, this.detections.length - DETECTIONS_CAP);
    return row;
  }

  private pushSpectrum(sensor: string, rf_mhz: number, levels: number[], now: number): void {
    // keep timestamps strictly increasing so "latest of any sensor" is well defined
    const t = Math.max(now, this.lastSpectrumTs + 1);
    this.lastSpectrumTs = t;
    const rows = this.spectra.get(sensor) ?? [];
    rows.push({ ts: new Date(t).toISOString(), t, rf_mhz, levels });
    if (rows.length > SPECTRA_CAP) rows.splice(0, rows.length - SPECTRA_CAP);
    this.spectra.set(sensor, rows);
  }

  /** One scenario tick: 3 COMINT captures, every second tick an R-ESM PDW batch, then fusion. */
  async step(): Promise<void> {
    if (this.busy || document.hidden) return;
    this.busy = true;
    try {
      const tick = this.tick;
      const rng = this.rng;
      const comint = SCENARIO.filter((e) => e.mod !== 'pulse');
      const drift = Math.sin(tick / 60); // slow bearing wander
      const groups: FusionDetection[][] = [];
      for (let s = 0; s < N_COMINT_SENSORS; s++) {
        const e = comint[(tick + s) % comint.length];
        const sid = `sirius-${s}`;
        const fr = this.assets.bank.draw(e.mod, rng);
        const L = fr.i.length;
        const i = new Float32Array(L);
        const q = new Float32Array(L);
        for (let k = 0; k < L; k++) {
          i[k] = fr.i[k] + rng.normal(0, FRAME_NOISE_SD);
          q[k] = fr.q[k] + rng.normal(0, FRAME_NOISE_SD);
        }
        const aoa = (((e.aoa + 3 * drift + rng.normal(0, 1.0)) % 360) + 360) % 360;
        const { probs, latencyMs } = await this.clf.classify(i, q);
        let best = 0;
        for (let k = 1; k < probs.length; k++) if (probs[k] > probs[best]) best = k;
        const label = this.clf.classes[best];
        const now = Date.now();
        this.counters.iq_total += 1;
        this.counters.accepted_total += 1;
        this.pushDetection(
          { sensor_id: sid, source: 'comint', rf_mhz: e.rf, aoa: round(aoa, 2), modulation: label, confidence: round(probs[best], 4), latency_ms: round(latencyMs, 3), pw_us: null, pri_us: null, pri_type: null },
          now,
        );
        this.pushSpectrum(sid, e.rf, welchPsd(i, q).db.map((v) => round(v, 2)), now);
        groups.push([{ rf_mhz: e.rf, aoa, source: 'comint', sensor_id: sid, label }]);
      }
      if (tick % 2 === 0) {
        const pdw = pdwBatch(rng, (tick * 1000) / (1000 / TICK_MS));
        this.counters.pdw_total += 1;
        this.counters.accepted_total += 1;
        if (pdw.length >= 5) {
          const t0 = performance.now();
          const { emitters } = deinterleavePdw(pdw);
          const dt = performance.now() - t0;
          const now = Date.now();
          const group: FusionDetection[] = [];
          for (const e of Object.values(emitters)) {
            const pri_us = e.pri_mean ? e.pri_mean * 1e3 : null;
            this.pushDetection(
              { sensor_id: ESM_SENSOR, source: 'resm', rf_mhz: round(e.rf, 3), aoa: round(e.aoa, 2), modulation: 'pulse', confidence: 1.0, latency_ms: round(dt, 3), pw_us: round(e.pw, 3), pri_us: pri_us === null ? null : round(pri_us, 1), pri_type: e.pri_type },
              now,
            );
            group.push({ rf_mhz: e.rf, aoa: e.aoa, source: 'resm', sensor_id: ESM_SENSOR, label: 'pulse', pw_us: e.pw, pri_us, pri_type: e.pri_type });
          }
          groups.push(group);
        }
      }
      this.tracks = this.fuser.stepMulti(groups, Date.now() / 1000);
      this.tick = tick + 1;
    } catch (e) {
      console.error('browser demo tick failed', e);
    } finally {
      this.busy = false;
    }
  }

  // ---------- queries used by the shim ----------

  recentDetections(minutes: number, limit: number, sensor?: string): DetectionRow[] {
    const since = Date.now() - minutes * 60_000;
    const out: DetectionRow[] = [];
    for (let k = this.detections.length - 1; k >= 0 && out.length < limit; k--) {
      const d = this.detections[k];
      if (d.t <= since) break;
      if (sensor && d.sensor_id !== sensor) continue;
      out.push(d);
    }
    return out;
  }

  addReport(question: string, report: string, thread: string): ReportRow {
    const row: ReportRow = { id: this.nextReportId++, ts: new Date().toISOString(), question, report, thread };
    this.reports.push(row);
    return row;
  }
}

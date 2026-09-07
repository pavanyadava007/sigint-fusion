// Browser-mode port of ml-service/fusion/associate.py: one Kalman track per emitter with state
// [rf, aoa, rf_rate, aoa_rate], gated Mahalanobis association (greedy nearest-within-gate instead
// of the Hungarian solver), bearing wrap, per-sensor groups and duplicate merging.

import { round } from './dsp';

export interface FusionDetection {
  rf_mhz: number;
  aoa: number;
  source: 'comint' | 'resm';
  sensor_id?: string;
  label?: string;
  pw_us?: number | null;
  pri_us?: number | null;
  pri_type?: string | null;
}

export interface EobTrack {
  track_id: number;
  rf_mhz: number;
  aoa: number;
  rf_rate: number;
  aoa_rate: number;
  sources: ('comint' | 'resm')[];
  sensors: string[];
  hits: number;
  misses: number;
  first_seen: string;
  last_seen: string;
  modulation: string | null;
  label_agreement: number | null;
  pw_us: number | null;
  pri_us: number | null;
  pri_type: string | null;
  updated: string;
}

type Mat = number[][];

const R_RF = 1.0; // measurement noise RF (MHz^2)
const R_AOA = 2.0; // measurement noise AOA (deg^2)

export function wrap(deg: number): number {
  return ((((deg + 180) % 360) + 360) % 360) - 180;
}

function eye(n: number, s = 1): Mat {
  return Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? s : 0)));
}

function matmul(a: Mat, b: Mat): Mat {
  const out: Mat = Array.from({ length: a.length }, () => new Array<number>(b[0].length).fill(0));
  for (let i = 0; i < a.length; i++) for (let k = 0; k < b.length; k++) for (let j = 0; j < b[0].length; j++) out[i][j] += a[i][k] * b[k][j];
  return out;
}

function transpose(a: Mat): Mat {
  return a[0].map((_, j) => a.map((row) => row[j]));
}

function add(a: Mat, b: Mat): Mat {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

function sub(a: Mat, b: Mat): Mat {
  return a.map((row, i) => row.map((v, j) => v - b[i][j]));
}

function inv2(s: Mat): Mat {
  const det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
  const d = det !== 0 ? det : 1e-12;
  return [
    [s[1][1] / d, -s[0][1] / d],
    [-s[1][0] / d, s[0][0] / d],
  ];
}

function isoSeconds(t: number): string {
  return new Date(t * 1000).toISOString();
}

let nextId = 1;

export class Track {
  x: number[];
  P: Mat;
  t: number;
  id: number;
  first_seen: number;
  sources = new Set<'comint' | 'resm'>();
  sensors = new Set<string>();
  hits = 0;
  misses = 0;
  labels: string[] = [];
  attrs: { pw_us?: number; pri_us?: number; pri_type?: string } = {};

  constructor(z: [number, number], det: FusionDetection, t: number) {
    this.x = [z[0], z[1], 0, 0];
    this.P = eye(4, 10);
    this.t = t;
    this.first_seen = t;
    this.id = nextId++;
    this.update(z, det, t);
  }

  predict(dt: number): void {
    const F = eye(4);
    F[0][2] = dt;
    F[1][3] = dt;
    this.x = [this.x[0] + dt * this.x[2], this.x[1] + dt * this.x[3], this.x[2], this.x[3]];
    this.P = add(matmul(matmul(F, this.P), transpose(F)), eye(4, 0.01));
  }

  innovation(z: [number, number]): { y: [number, number]; S: Mat } {
    const S: Mat = [
      [this.P[0][0] + R_RF, this.P[0][1]],
      [this.P[1][0], this.P[1][1] + R_AOA],
    ];
    const y: [number, number] = [z[0] - this.x[0], wrap(z[1] - this.x[1])];
    return { y, S };
  }

  mahalanobis(z: [number, number]): number {
    const { y, S } = this.innovation(z);
    const Si = inv2(S);
    const q = y[0] * (Si[0][0] * y[0] + Si[0][1] * y[1]) + y[1] * (Si[1][0] * y[0] + Si[1][1] * y[1]);
    return Math.sqrt(Math.max(q, 0));
  }

  update(z: [number, number], det: FusionDetection, t: number): void {
    const { y, S } = this.innovation(z);
    const Si = inv2(S);
    // K = P H^T S^-1, H = [I2 0] -> P H^T is the first two columns of P
    const PHt: Mat = this.P.map((row) => [row[0], row[1]]);
    const K = matmul(PHt, Si);
    for (let i = 0; i < 4; i++) this.x[i] += K[i][0] * y[0] + K[i][1] * y[1];
    const KH: Mat = K.map((row) => [row[0], row[1], 0, 0]);
    this.P = matmul(sub(eye(4), KH), this.P);
    this.t = t;
    this.hits += 1;
    this.misses = 0;
    this.sources.add(det.source);
    if (det.sensor_id) this.sensors.add(det.sensor_id);
    if (det.label) this.labels.push(det.label);
    if (det.pw_us !== undefined && det.pw_us !== null) this.attrs.pw_us = det.pw_us;
    if (det.pri_us !== undefined && det.pri_us !== null) this.attrs.pri_us = det.pri_us;
    if (det.pri_type !== undefined && det.pri_type !== null) this.attrs.pri_type = det.pri_type;
  }

  eob(now: number): EobTrack {
    const counts = new Map<string, number>();
    for (const l of this.labels) counts.set(l, (counts.get(l) ?? 0) + 1);
    let best: [string, number] | null = null;
    for (const [k, v] of counts) if (!best || v > best[1]) best = [k, v];
    return {
      track_id: this.id,
      rf_mhz: round(this.x[0], 2),
      aoa: round(((this.x[1] % 360) + 360) % 360, 1),
      rf_rate: round(this.x[2], 3),
      aoa_rate: round(this.x[3], 3),
      sources: Array.from(this.sources).sort(),
      sensors: Array.from(this.sensors).sort(),
      hits: this.hits,
      misses: this.misses,
      first_seen: isoSeconds(this.first_seen),
      last_seen: isoSeconds(this.t),
      modulation: best ? best[0] : null,
      label_agreement: best ? round(best[1] / this.labels.length, 2) : null,
      pw_us: this.attrs.pw_us ?? null,
      pri_us: this.attrs.pri_us ?? null,
      pri_type: this.attrs.pri_type ?? null,
      updated: isoSeconds(now),
    };
  }
}

interface Merged extends FusionDetection {
  _n: number;
  _labels: (string | undefined)[];
}

/** Merge repeated observations of one emitter inside one detection set (same source, RF within
 *  max(2 MHz, 0.5 %), bearing within aoaTol): RF/AOA averaged, most frequent label kept. */
export function mergeDuplicates(detections: FusionDetection[], rfTol = 0.005, aoaTol = 5): FusionDetection[] {
  const out: Merged[] = [];
  for (const d of [...detections].sort((a, b) => a.rf_mhz - b.rf_mhz)) {
    let merged = false;
    for (const m of out) {
      if (m.source === d.source && Math.abs(m.rf_mhz - d.rf_mhz) <= Math.max(2, rfTol * m.rf_mhz) && Math.abs(wrap(m.aoa - d.aoa)) <= aoaTol) {
        const n = m._n;
        m.rf_mhz = (m.rf_mhz * n + d.rf_mhz) / (n + 1);
        m.aoa = (((m.aoa + wrap(d.aoa - m.aoa) / (n + 1)) % 360) + 360) % 360;
        m._labels.push(d.label);
        m._n = n + 1;
        merged = true;
        break;
      }
    }
    if (!merged) out.push({ ...d, _n: 1, _labels: [d.label] });
  }
  return out.map((m) => {
    const labs = m._labels.filter((x): x is string => !!x);
    const rest: FusionDetection = { ...m };
    delete (rest as Partial<Merged>)._n;
    delete (rest as Partial<Merged>)._labels;
    if (labs.length) {
      const counts = new Map<string, number>();
      for (const l of labs) counts.set(l, (counts.get(l) ?? 0) + 1);
      rest.label = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0][0];
    }
    return rest;
  });
}

export class Fuser {
  tracks: Track[] = [];
  t = 0;
  constructor(
    public gate = 4.0,
    public maxMisses = 20,
  ) {}

  step(detections: FusionDetection[], t: number): EobTrack[] {
    return this.stepMulti([detections], t);
  }

  /** One time step with several independent detection sets (one per sensor). */
  stepMulti(groups: FusionDetection[][], t: number): EobTrack[] {
    const dt = Math.max(t - this.t, 0);
    this.t = t;
    for (const tr of this.tracks) tr.predict(dt);
    const updated = new Set<Track>();
    for (const dets of groups) for (const tr of this.associate(mergeDuplicates(dets), t)) updated.add(tr);
    for (const tr of this.tracks) if (!updated.has(tr) && tr.t !== t) tr.misses += 1;
    this.tracks = this.tracks.filter((tr) => tr.misses <= this.maxMisses);
    return this.eob();
  }

  private associate(detections: FusionDetection[], t: number): Track[] {
    const updated: Track[] = [];
    if (!detections.length) return updated;
    const Z: [number, number][] = detections.map((d) => [d.rf_mhz, d.aoa]);
    const matched = new Set<number>();
    if (this.tracks.length) {
      const C = this.tracks.map((tr) => Z.map((z) => tr.mahalanobis(z)));
      const usedTracks = new Set<number>();
      // greedy: repeatedly take the globally closest (track, detection) pair inside the gate
      for (;;) {
        let bi = -1;
        let bj = -1;
        let best = this.gate;
        for (let i = 0; i < C.length; i++) {
          if (usedTracks.has(i)) continue;
          for (let j = 0; j < Z.length; j++) {
            if (matched.has(j)) continue;
            if (C[i][j] < best) {
              best = C[i][j];
              bi = i;
              bj = j;
            }
          }
        }
        if (bi < 0) break;
        this.tracks[bi].update(Z[bj], detections[bj], t);
        updated.push(this.tracks[bi]);
        usedTracks.add(bi);
        matched.add(bj);
      }
    }
    for (let j = 0; j < Z.length; j++) {
      if (matched.has(j)) continue;
      const tr = new Track(Z[j], detections[j], t);
      this.tracks.push(tr);
      updated.push(tr);
    }
    return updated;
  }

  eob(): EobTrack[] {
    const now = Date.now() / 1000;
    return this.tracks.map((tr) => tr.eob(now)).sort((a, b) => b.hits - a.hits || a.track_id - b.track_id);
  }
}

// Browser-mode port of ml-service/sigproc/dsp.py: Welch PSD, CA-CFAR, STFT spectrogram and
// DBSCAN pulse deinterleaving. Shapes and scaling follow scipy so the pages see the same numbers
// as they would from the server (256 PSD bins, 64 spectrogram bins, dB with a 1e-12 floor).

const EPS_DB = 1e-12;

// ---------- FFT ----------

/** In-place complex FFT. Radix-2 for powers of two, plain DFT otherwise (n <= 256 in practice). */
export function fft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) !== 0) {
    dft(re, im);
    return;
  }
  // bit reversal
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      const half = len >> 1;
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const xr = re[b] * cr - im[b] * ci;
        const xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

function dft(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  const or = Float64Array.from(re);
  const oi = Float64Array.from(im);
  for (let k = 0; k < n; k++) {
    let sr = 0;
    let si = 0;
    for (let t = 0; t < n; t++) {
      const ang = (-2 * Math.PI * k * t) / n;
      const c = Math.cos(ang);
      const s = Math.sin(ang);
      sr += or[t] * c - oi[t] * s;
      si += or[t] * s + oi[t] * c;
    }
    re[k] = sr;
    im[k] = si;
  }
}

/** Periodic Hann window (scipy get_window('hann', n) default, sym=False). */
export function hann(n: number): Float64Array {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/** fftshift(fftfreq(n, 1/fs)): ascending bin centre frequencies. */
export function shiftedFreqs(n: number, fs = 1): number[] {
  const out: number[] = new Array(n);
  const half = Math.floor(n / 2);
  for (let i = 0; i < n; i++) out[i] = ((i - half) * fs) / n;
  return out;
}

/** Index in the unshifted FFT output that lands at shifted position i. */
function unshiftIndex(i: number, n: number): number {
  const half = Math.floor(n / 2);
  return (i + n - half) % n;
}

// ---------- Welch PSD ----------

export interface Psd {
  freqs: number[];
  db: number[];
}

/**
 * Welch PSD of complex baseband (i, q), scipy defaults: nperseg = min(256, L), 50 % overlap,
 * periodic Hann, constant detrend per segment, density scaling, two-sided, fftshift, 10log10.
 */
export function welchPsd(i: ArrayLike<number>, q: ArrayLike<number>, fs = 1, nperseg = 256): Psd {
  const L = i.length;
  const n = Math.min(nperseg, L);
  const step = n - Math.floor(n / 2);
  const w = hann(n);
  let wsq = 0;
  for (let k = 0; k < n; k++) wsq += w[k] * w[k];
  const scale = 1 / (fs * wsq);
  const nseg = Math.floor((L - n) / step) + 1;
  const acc = new Float64Array(n);
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let s = 0; s < nseg; s++) {
    const off = s * step;
    let mr = 0;
    let mi = 0;
    for (let k = 0; k < n; k++) {
      mr += i[off + k];
      mi += q[off + k];
    }
    mr /= n;
    mi /= n;
    for (let k = 0; k < n; k++) {
      re[k] = (i[off + k] - mr) * w[k];
      im[k] = (q[off + k] - mi) * w[k];
    }
    fft(re, im);
    for (let k = 0; k < n; k++) acc[k] += (re[k] * re[k] + im[k] * im[k]) * scale;
  }
  const db: number[] = new Array(n);
  for (let k = 0; k < n; k++) db[k] = 10 * Math.log10(acc[unshiftIndex(k, n)] / Math.max(nseg, 1) + EPS_DB);
  return { freqs: shiftedFreqs(n, fs), db };
}

// ---------- CA-CFAR ----------

/** 1-D cell-averaging CFAR on a dB vector; edges use the training cells that exist. */
export function caCfar(xDb: ArrayLike<number>, guard = 2, train = 8, pfa = 1e-3): boolean[] {
  const n = xDb.length;
  const lin = new Float64Array(n);
  for (let k = 0; k < n; k++) lin[k] = Math.pow(10, xDb[k] / 10);
  const ntrain = 2 * train;
  const alpha = ntrain * (Math.pow(pfa, -1 / ntrain) - 1);
  const cs = new Float64Array(n + 1);
  for (let k = 0; k < n; k++) cs[k + 1] = cs[k] + lin[k];
  const clip = (v: number) => Math.max(0, Math.min(n, v));
  const mask: boolean[] = new Array(n);
  for (let k = 0; k < n; k++) {
    const lo = clip(k - guard - train);
    const loEnd = clip(k - guard);
    const hiStart = clip(k + guard + 1);
    const hi = clip(k + guard + train + 1);
    const total = cs[loEnd] - cs[lo] + (cs[hi] - cs[hiStart]);
    const count = loEnd - lo + (hi - hiStart);
    const mean = count > 0 ? total / count : Infinity;
    mask[k] = lin[k] > alpha * mean;
  }
  return mask;
}

// ---------- STFT spectrogram ----------

export interface Stft {
  freqs: number[];
  times: number[];
  db: number[][]; // [freq][time]
}

/**
 * scipy.signal.stft semantics: nperseg 64, noverlap 32, periodic Hann, boundary='zeros'
 * (nperseg/2 zeros each side), padded to a whole number of hops, scaling='spectrum' (1/sum(w)),
 * two-sided, fftshift on the frequency axis, 20log10 magnitude.
 */
export function stft(i: ArrayLike<number>, q: ArrayLike<number>, fs = 1, nperseg = 64): Stft {
  const L = i.length;
  const hop = nperseg - Math.floor(nperseg / 2);
  const pad = Math.floor(nperseg / 2);
  let total = L + 2 * pad;
  const rem = (total - nperseg) % hop;
  if (rem !== 0) total += hop - rem;
  const xr = new Float64Array(total);
  const xi = new Float64Array(total);
  for (let k = 0; k < L; k++) {
    xr[pad + k] = i[k];
    xi[pad + k] = q[k];
  }
  const w = hann(nperseg);
  let wsum = 0;
  for (let k = 0; k < nperseg; k++) wsum += w[k];
  const scale = 1 / wsum;
  const nT = Math.floor((total - nperseg) / hop) + 1;
  const db: number[][] = [];
  for (let f = 0; f < nperseg; f++) db.push(new Array<number>(nT));
  const re = new Float64Array(nperseg);
  const im = new Float64Array(nperseg);
  const times: number[] = new Array(nT);
  for (let t = 0; t < nT; t++) {
    const off = t * hop;
    for (let k = 0; k < nperseg; k++) {
      re[k] = xr[off + k] * w[k];
      im[k] = xi[off + k] * w[k];
    }
    fft(re, im);
    for (let f = 0; f < nperseg; f++) {
      const u = unshiftIndex(f, nperseg);
      db[f][t] = 20 * Math.log10(Math.hypot(re[u], im[u]) * scale + EPS_DB);
    }
    times[t] = (t * hop) / fs;
  }
  return { freqs: shiftedFreqs(nperseg, fs), times, db };
}

// ---------- PDW deinterleaving (DBSCAN) ----------

export interface EmitterStats {
  n: number;
  rf: number;
  pw: number;
  aoa: number;
  pri_mean: number | null;
  pri_type: string;
}

function std(v: number[]): number {
  if (!v.length) return 0;
  const m = v.reduce((a, b) => a + b, 0) / v.length;
  return Math.sqrt(v.reduce((a, b) => a + (b - m) * (b - m), 0) / v.length);
}

function mean(v: number[]): number {
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
}

function dbscan(feats: number[][], eps: number, minSamples: number): number[] {
  const n = feats.length;
  const labels = new Array<number>(n).fill(-2); // -2 unvisited, -1 noise
  const eps2 = eps * eps;
  const neighbours = (p: number): number[] => {
    const out: number[] = [];
    const a = feats[p];
    for (let j = 0; j < n; j++) {
      const b = feats[j];
      let d = 0;
      for (let k = 0; k < a.length; k++) d += (a[k] - b[k]) * (a[k] - b[k]);
      if (d <= eps2) out.push(j);
    }
    return out;
  };
  let cluster = 0;
  for (let p = 0; p < n; p++) {
    if (labels[p] !== -2) continue;
    const nb = neighbours(p);
    if (nb.length < minSamples) {
      labels[p] = -1;
      continue;
    }
    labels[p] = cluster;
    const queue = nb.slice();
    for (let qi = 0; qi < queue.length; qi++) {
      const r = queue[qi];
      if (labels[r] === -1) labels[r] = cluster;
      if (labels[r] !== -2) continue;
      labels[r] = cluster;
      const nb2 = neighbours(r);
      if (nb2.length >= minSamples) for (const x of nb2) if (labels[x] === -2 || labels[x] === -1) queue.push(x);
    }
    cluster++;
  }
  return labels;
}

function priType(pri: number[], rf: number[]): string {
  if (pri.length < 3) return 'unknown';
  if (new Set(rf.map((v) => Math.round(v / 50))).size >= 3) return 'agile';
  const m = mean(pri);
  const cv = std(pri) / (m + 1e-9);
  if (cv < 0.05) return 'constant';
  const levels = new Set(pri.map((v) => Math.round(v / (m * 0.05)))).size;
  return levels <= 6 ? 'stagger' : 'jitter';
}

/**
 * pdw rows [toa_ms, rf_mhz, pw_us, aoa_deg]. DBSCAN (eps 0.15, min_samples 5) on RF, PW, AOA each
 * divided by its standard deviation, then per-cluster means and PRI statistics from sorted TOA diffs.
 */
export function deinterleavePdw(pdw: number[][], eps = 0.15, minSamples = 5): { labels: number[]; emitters: Record<string, EmitterStats> } {
  if (!pdw.length) return { labels: [], emitters: {} };
  const cols = [1, 2, 3].map((c) => pdw.map((r) => r[c]));
  const sd = cols.map((c) => std(c) + 1e-9);
  const feats = pdw.map((r) => [r[1] / sd[0], r[2] / sd[1], r[3] / sd[2]]);
  const labels = dbscan(feats, eps, minSamples);
  const emitters: Record<string, EmitterStats> = {};
  const ids = Array.from(new Set(labels.filter((l) => l >= 0))).sort((a, b) => a - b);
  for (const lab of ids) {
    const rows = pdw.filter((_, k) => labels[k] === lab);
    const toa = rows.map((r) => r[0]).sort((a, b) => a - b);
    const pri: number[] = [];
    for (let k = 1; k < toa.length; k++) pri.push(toa[k] - toa[k - 1]);
    const rf = rows.map((r) => r[1]);
    emitters[String(lab)] = {
      n: rows.length,
      rf: mean(rf),
      pw: mean(rows.map((r) => r[2])),
      aoa: mean(rows.map((r) => r[3])),
      pri_mean: pri.length ? mean(pri) : null,
      pri_type: priType(pri, rf),
    };
  }
  return { labels, emitters };
}

// ---------- helpers shared by the shim ----------

export function round(v: number, digits: number): number {
  const f = Math.pow(10, digits);
  return Math.round(v * f) / f;
}

export function softmax(logits: ArrayLike<number>): number[] {
  let mx = -Infinity;
  for (let k = 0; k < logits.length; k++) if (logits[k] > mx) mx = logits[k];
  const e: number[] = new Array(logits.length);
  let s = 0;
  for (let k = 0; k < logits.length; k++) {
    e[k] = Math.exp(logits[k] - mx);
    s += e[k];
  }
  return e.map((v) => v / s);
}

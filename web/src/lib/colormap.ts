// Viridis-like perceptual colormap (10 anchor stops from matplotlib's viridis,
// linearly interpolated in sRGB). Used for the waterfall and spectrogram, which
// need a high-dynamic-range sequential scale; a scale bar is always shown next
// to it so the mapping is never colour-alone.
const STOPS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 73, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [110, 206, 88],
  [181, 222, 43],
  [253, 231, 37],
];

const LUT_N = 256;
const LUT = new Uint8ClampedArray(LUT_N * 3);
for (let i = 0; i < LUT_N; i++) {
  const t = (i / (LUT_N - 1)) * (STOPS.length - 1);
  const k = Math.min(Math.floor(t), STOPS.length - 2);
  const f = t - k;
  const a = STOPS[k];
  const b = STOPS[k + 1];
  LUT[i * 3] = a[0] + (b[0] - a[0]) * f;
  LUT[i * 3 + 1] = a[1] + (b[1] - a[1]) * f;
  LUT[i * 3 + 2] = a[2] + (b[2] - a[2]) * f;
}

/** Write the colour for t in [0,1] into an RGBA buffer at byte offset o. */
export function viridisInto(buf: Uint8ClampedArray, o: number, t: number): void {
  const idx = t <= 0 ? 0 : t >= 1 ? LUT_N - 1 : Math.round(t * (LUT_N - 1));
  buf[o] = LUT[idx * 3];
  buf[o + 1] = LUT[idx * 3 + 1];
  buf[o + 2] = LUT[idx * 3 + 2];
  buf[o + 3] = 255;
}

export function viridisCss(t: number): string {
  const idx = Math.round(Math.min(1, Math.max(0, t)) * (LUT_N - 1));
  return `rgb(${LUT[idx * 3]},${LUT[idx * 3 + 1]},${LUT[idx * 3 + 2]})`;
}

/** Robust dB range: 2nd to 98th percentile of a sample of values. */
export function robustRange(values: Float32Array | number[], lo = 0.02, hi = 0.98): [number, number] {
  const n = values.length;
  if (n === 0) return [-100, 0];
  const step = Math.max(1, Math.floor(n / 4096));
  const sample: number[] = [];
  for (let i = 0; i < n; i += step) {
    const v = values[i];
    if (Number.isFinite(v)) sample.push(v);
  }
  if (sample.length === 0) return [-100, 0];
  sample.sort((a, b) => a - b);
  const a = sample[Math.floor(lo * (sample.length - 1))];
  const b = sample[Math.floor(hi * (sample.length - 1))];
  return b - a < 1e-6 ? [a - 1, a + 1] : [a, b];
}

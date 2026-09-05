import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import type { SpectrogramResult } from '../api';
import { useElementSize } from '../hooks/useResize';
import { useThemeVersion } from '../hooks/useTheme';
import { robustRange, viridisInto } from '../lib/colormap';
import { fmtNum } from '../lib/format';
import { chartTokens, setupCanvas } from '../lib/theme';
import { ScaleBar } from './ScaleBar';

const L = 54;
const R = 12;
const T = 8;
const B = 30;

function fmtTick(v: number): string {
  if (Math.abs(v) < 5e-3) v = 0; // avoid a signed zero tick
  const a = Math.abs(v);
  if (a >= 1e6) return `${fmtNum(v / 1e6, 2)}M`;
  if (a >= 1e3) return `${fmtNum(v / 1e3, 1)}k`;
  if (a >= 10) return fmtNum(v, 0);
  return fmtNum(v, 2);
}

/** STFT spectrogram: x = time frame, y = frequency bin (low at bottom). */
export function SpectrogramCanvas({ data, height = 280 }: { data: SpectrogramResult; height?: number }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrapRef);
  const themeV = useThemeVersion();
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);

  const nF = data.db.length;
  const nT = nF ? data.db[0].length : 0;
  const ascending = data.freqs.length > 1 ? data.freqs[0] < data.freqs[data.freqs.length - 1] : true;

  const image = useMemo(() => {
    if (!nF || !nT) return null;
    const all = new Float32Array(nF * nT);
    for (let f = 0; f < nF; f++) for (let t = 0; t < nT; t++) all[f * nT + t] = data.db[f][t] ?? NaN;
    const range = robustRange(all, 0.01, 0.995);
    const span = range[1] - range[0] || 1;
    const buf = new Uint8ClampedArray(nF * nT * 4);
    for (let f = 0; f < nF; f++) {
      const row = ascending ? nF - 1 - f : f; // put low frequency at the bottom
      for (let t = 0; t < nT; t++) {
        const v = all[f * nT + t];
        viridisInto(buf, (row * nT + t) * 4, Number.isFinite(v) ? (v - range[0]) / span : 0);
      }
    }
    const off = document.createElement('canvas');
    off.width = nT;
    off.height = nF;
    off.getContext('2d')?.putImageData(new ImageData(buf, nT, nF), 0, 0);
    return { off, range };
  }, [data, nF, nT, ascending]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || width < 10 || !image) return;
    const ctx = setupCanvas(c, width, height);
    if (!ctx) return;
    const tk = chartTokens();
    ctx.fillStyle = tk.surface;
    ctx.fillRect(0, 0, width, height);
    const pw = width - L - R;
    const ph = height - T - B;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image.off, L, T, pw, ph);
    ctx.strokeStyle = tk.axis;
    ctx.lineWidth = 1;
    ctx.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);
    ctx.font = `10px ${tk.mono}`;
    ctx.fillStyle = tk.muted;
    // y ticks: frequency
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const fmin = Math.min(data.freqs[0], data.freqs[data.freqs.length - 1]);
    const fmax = Math.max(data.freqs[0], data.freqs[data.freqs.length - 1]);
    for (let k = 0; k <= 4; k++) {
      const y = T + ph - (k / 4) * ph;
      ctx.fillText(fmtTick(fmin + ((fmax - fmin) * k) / 4), L - 5, y);
    }
    ctx.save();
    ctx.translate(12, T + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('frequency', 0, 0);
    ctx.restore();
    // x ticks: time
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const tmin = data.times[0] ?? 0;
    const tmax = data.times[data.times.length - 1] ?? nT - 1;
    for (let k = 0; k <= 4; k++) {
      const x = L + (k / 4) * pw;
      ctx.fillText(fmtTick(tmin + ((tmax - tmin) * k) / 4), x, T + ph + 5);
    }
    ctx.fillText('time', L + pw / 2, T + ph + 17);
  }, [image, width, height, themeV, data, nT]);

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!image) return;
    const x = e.nativeEvent.offsetX;
    const y = e.nativeEvent.offsetY;
    const pw = width - L - R;
    const ph = height - T - B;
    if (x < L || x > L + pw || y < T || y > T + ph) return setHover(null);
    const t = Math.min(nT - 1, Math.floor(((x - L) / pw) * nT));
    const row = Math.min(nF - 1, Math.floor(((y - T) / ph) * nF));
    const f = ascending ? nF - 1 - row : row;
    setHover({ x, y, text: `t ${fmtTick(data.times[t] ?? t)}  f ${fmtTick(data.freqs[f] ?? f)}  ${fmtNum(data.db[f][t], 1)} dB` });
  };

  return (
    <div>
      <div className="canvas-wrap" ref={wrapRef} style={{ height }}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`Spectrogram, ${nF} frequency bins by ${nT} time frames`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
        {hover && (
          <div className="tooltip" style={{ left: Math.min(hover.x + 12, Math.max(0, width - 220)), top: Math.max(0, hover.y - 30) }}>
            {hover.text}
          </div>
        )}
      </div>
      <div className="legend" style={{ justifyContent: 'flex-end' }}>
        {image && <ScaleBar min={image.range[0]} max={image.range[1]} />}
      </div>
    </div>
  );
}

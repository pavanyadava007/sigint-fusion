import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import type { SpectrumRow } from '../api';
import { useElementSize } from '../hooks/useResize';
import { useThemeVersion } from '../hooks/useTheme';
import { robustRange, viridisInto } from '../lib/colormap';
import { fmtNum, fmtTime } from '../lib/format';
import { chartTokens, setupCanvas } from '../lib/theme';
import { ScaleBar } from './ScaleBar';

const L = 58;
const R = 14;
const T = 8;
const B = 34;

interface Props {
  rows: SpectrumRow[]; // oldest first
  centreMhz: number | null;
  height?: number;
}

/**
 * Spectrum waterfall. Rows are PSD sweeps (oldest at top, newest at bottom);
 * columns are FFT bins. The contract does not carry a sample rate, so the
 * frequency axis is labelled as bin offset from the centre bin, i.e. relative
 * to the sensor's centre frequency fc, spanning -fs/2 .. +fs/2.
 */
export function Waterfall({ rows, centreMhz, height = 340 }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrapRef);
  const themeV = useThemeVersion();
  const rangeRef = useRef<[number, number] | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);

  const nRows = rows.length;
  const nBins = nRows ? rows[0].levels.length : 0;

  const image = useMemo(() => {
    if (!nRows || !nBins) return null;
    const all = new Float32Array(nRows * nBins);
    for (let r = 0; r < nRows; r++) {
      const lv = rows[r].levels;
      for (let b = 0; b < nBins; b++) all[r * nBins + b] = lv[b] ?? NaN;
    }
    const target = robustRange(all);
    const prev = rangeRef.current;
    // Ease the colour range so a single hot sweep does not flash the whole picture.
    const range: [number, number] = prev
      ? [prev[0] + (target[0] - prev[0]) * 0.3, prev[1] + (target[1] - prev[1]) * 0.3]
      : target;
    rangeRef.current = range;
    const span = range[1] - range[0] || 1;
    const buf = new Uint8ClampedArray(nRows * nBins * 4);
    for (let k = 0; k < nRows * nBins; k++) {
      const v = all[k];
      viridisInto(buf, k * 4, Number.isFinite(v) ? (v - range[0]) / span : 0);
    }
    const off = document.createElement('canvas');
    off.width = nBins;
    off.height = nRows;
    off.getContext('2d')?.putImageData(new ImageData(buf, nBins, nRows), 0, 0);
    return { off, range };
  }, [rows, nRows, nBins]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || width < 10) return;
    const ctx = setupCanvas(c, width, height);
    if (!ctx) return;
    const tk = chartTokens();
    ctx.fillStyle = tk.surface;
    ctx.fillRect(0, 0, width, height);
    const pw = width - L - R;
    const ph = height - T - B;
    ctx.font = `10px ${tk.mono}`;
    ctx.fillStyle = tk.muted;
    ctx.strokeStyle = tk.axis;
    ctx.lineWidth = 1;

    if (image) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(image.off, L, T, pw, ph);
    } else {
      ctx.fillStyle = tk.grid;
      ctx.fillRect(L, T, pw, ph);
      ctx.fillStyle = tk.muted;
    }
    // plot frame
    ctx.strokeRect(L + 0.5, T + 0.5, pw - 1, ph - 1);

    // time axis (left): oldest at top, newest at bottom
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = tk.muted;
    if (nRows) {
      const ticks = nRows >= 4 ? [0, Math.floor(nRows / 3), Math.floor((2 * nRows) / 3), nRows - 1] : [0, nRows - 1];
      for (const r of new Set(ticks)) {
        const y = T + ((r + 0.5) / nRows) * ph;
        ctx.fillText(fmtTime(rows[r].ts), L - 6, y);
        ctx.beginPath();
        ctx.moveTo(L - 3, y + 0.5);
        ctx.lineTo(L, y + 0.5);
        ctx.stroke();
      }
    } else {
      ctx.fillText('t', L - 6, T + ph / 2);
    }

    // frequency axis (bottom), relative to fc
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const fr = [
      [0, '-fs/2'],
      [0.25, '-fs/4'],
      [0.5, 'fc'],
      [0.75, '+fs/4'],
      [1, '+fs/2'],
    ] as const;
    for (const [f, label] of fr) {
      const x = L + f * pw;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, T + ph);
      ctx.lineTo(Math.round(x) + 0.5, T + ph + 4);
      ctx.stroke();
      ctx.fillText(label, x, T + ph + 6);
    }
    // centre marker line
    ctx.strokeStyle = tk.ink2;
    ctx.globalAlpha = 0.35;
    ctx.beginPath();
    ctx.moveTo(Math.round(L + pw / 2) + 0.5, T);
    ctx.lineTo(Math.round(L + pw / 2) + 0.5, T + ph);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = tk.muted;
    ctx.textAlign = 'left';
    const caption =
      centreMhz !== null
        ? `fc = ${fmtNum(centreMhz, 1)} MHz, ${nBins} bins, offset from centre`
        : 'centre frequency unknown';
    ctx.fillText(caption, L, T + ph + 19);
  }, [image, width, height, themeV, rows, nRows, nBins, centreMhz]);

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!image || !nRows) return setHover(null);
    const x = e.nativeEvent.offsetX;
    const y = e.nativeEvent.offsetY;
    const pw = width - L - R;
    const ph = height - T - B;
    if (x < L || x > L + pw || y < T || y > T + ph) return setHover(null);
    const bin = Math.min(nBins - 1, Math.floor(((x - L) / pw) * nBins));
    const row = Math.min(nRows - 1, Math.floor(((y - T) / ph) * nRows));
    const v = rows[row].levels[bin];
    const off = bin - Math.floor(nBins / 2);
    setHover({
      x,
      y,
      text: `${fmtTime(rows[row].ts)}  bin ${off >= 0 ? '+' : ''}${off}  ${fmtNum(v, 1)} dB`,
    });
  };

  return (
    <div>
      <div className="canvas-wrap" ref={wrapRef} style={{ height }}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={
            nRows ? `Spectrum waterfall, ${nRows} sweeps of ${nBins} bins` : 'Spectrum waterfall, no sweeps received'
          }
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        />
        {hover && (
          <div
            className="tooltip"
            style={{ left: Math.min(hover.x + 12, Math.max(0, width - 220)), top: Math.max(0, hover.y - 30) }}
          >
            {hover.text}
          </div>
        )}
      </div>
      <div className="legend" style={{ justifyContent: 'space-between' }}>
        <span className="muted mono">older at top, newest at bottom</span>
        {image && <ScaleBar min={image.range[0]} max={image.range[1]} />}
      </div>
    </div>
  );
}

import { useEffect, useRef } from 'react';
import type { IQ } from '../api';
import { useElementSize } from '../hooks/useResize';
import { useThemeVersion } from '../hooks/useTheme';
import { fmtNum } from '../lib/format';
import { chartTokens, setupCanvas } from '../lib/theme';

export function Constellation({ iq, height = 280 }: { iq: IQ; height?: number }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrapRef);
  const themeV = useThemeVersion();

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || width < 10) return;
    const ctx = setupCanvas(c, width, height);
    if (!ctx) return;
    const tk = chartTokens();
    ctx.fillStyle = tk.surface;
    ctx.fillRect(0, 0, width, height);
    const n = Math.min(iq.i.length, iq.q.length);
    let m = 0;
    for (let k = 0; k < n; k++) m = Math.max(m, Math.abs(iq.i[k]), Math.abs(iq.q[k]));
    if (!Number.isFinite(m) || m === 0) m = 1;
    const lim = m * 1.1;
    const side = Math.min(width, height) - 30;
    const cx = width / 2;
    const cy = height / 2;
    const sx = (v: number) => cx + (v / lim) * (side / 2);
    const sy = (v: number) => cy - (v / lim) * (side / 2);

    ctx.strokeStyle = tk.grid;
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(cx - side / 2) + 0.5, Math.round(cy - side / 2) + 0.5, side, side);
    ctx.strokeStyle = tk.axis;
    ctx.beginPath();
    ctx.moveTo(cx - side / 2, Math.round(cy) + 0.5);
    ctx.lineTo(cx + side / 2, Math.round(cy) + 0.5);
    ctx.moveTo(Math.round(cx) + 0.5, cy - side / 2);
    ctx.lineTo(Math.round(cx) + 0.5, cy + side / 2);
    ctx.stroke();
    ctx.font = `10px ${tk.mono}`;
    ctx.fillStyle = tk.muted;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('I', cx + side / 2 - 6, cy + 3);
    ctx.fillText(`+${fmtNum(lim, 2)}`, cx + side / 2, cy + side / 2 + 4);
    ctx.fillText(`-${fmtNum(lim, 2)}`, cx - side / 2, cy + side / 2 + 4);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('Q', cx + 4, cy - side / 2 + 6);

    ctx.fillStyle = tk.s1;
    ctx.globalAlpha = n > 2048 ? 0.35 : 0.65;
    for (let k = 0; k < n; k++) {
      const x = sx(iq.i[k]);
      const y = sy(iq.q[k]);
      ctx.fillRect(x - 1.25, y - 1.25, 2.5, 2.5);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = tk.muted;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(`${n} samples`, width - 6, 4);
  }, [iq, width, height, themeV]);

  return (
    <div className="canvas-wrap" ref={wrapRef} style={{ height }}>
      <canvas ref={canvasRef} role="img" aria-label={`Constellation of ${Math.min(iq.i.length, iq.q.length)} I/Q samples`} />
    </div>
  );
}

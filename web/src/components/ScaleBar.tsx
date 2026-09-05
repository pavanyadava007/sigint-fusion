import { useEffect, useRef } from 'react';
import { viridisCss } from '../lib/colormap';
import { fmtNum } from '../lib/format';

/** Colour scale legend for the viridis heatmaps: a chart is never colour-alone. */
export function ScaleBar({ min, max, unit = 'dB' }: { min: number; max: number; unit?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = 120;
    c.height = 8;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    for (let x = 0; x < 120; x++) {
      ctx.fillStyle = viridisCss(x / 119);
      ctx.fillRect(x, 0, 1, 8);
    }
  }, []);
  return (
    <div className="scalebar" aria-label={`colour scale ${fmtNum(min, 0)} to ${fmtNum(max, 0)} ${unit}`}>
      <span>{fmtNum(min, 0)} {unit}</span>
      <canvas ref={ref} aria-hidden />
      <span>{fmtNum(max, 0)} {unit}</span>
    </div>
  );
}

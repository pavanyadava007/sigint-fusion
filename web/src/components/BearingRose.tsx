import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import type { Track } from '../api';
import { useElementSize } from '../hooks/useResize';
import { useThemeVersion } from '../hooks/useTheme';
import { fmtDeg, fmtFreqShort } from '../lib/format';
import { chartTokens, setupCanvas } from '../lib/theme';

interface Props {
  tracks: Track[];
  selectedId?: number | null;
  onSelect?: (id: number | null) => void;
  height?: number;
  compact?: boolean;
}

type Cat = 'comint' | 'resm' | 'both';
function catOf(t: Track): Cat {
  const c = t.sources.includes('comint');
  const r = t.sources.includes('resm');
  return c && r ? 'both' : r ? 'resm' : 'comint';
}

/**
 * Bearing rose: angle = AOA (0 deg at top, clockwise), radius = log10(RF).
 * Colour = source category (fixed slots: comint s1, resm s2, both s3);
 * marker area grows mildly with hit count. Legend below, tooltip on hover.
 */
export function BearingRose({ tracks, selectedId = null, onSelect, height = 320, compact = false }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { width } = useElementSize(wrapRef);
  const themeV = useThemeVersion();
  const [hover, setHover] = useState<{ x: number; y: number; text: string } | null>(null);
  const posRef = useRef<{ id: number; x: number; y: number; r: number }[]>([]);

  const scale = useMemo(() => {
    const rfs = tracks.map((t) => t.rf_mhz).filter((v) => Number.isFinite(v) && v > 0);
    if (!rfs.length) return { lo: 1, hi: 5 }; // 10 MHz .. 100 GHz
    let lo = Math.floor(Math.log10(Math.min(...rfs)) * 2) / 2;
    let hi = Math.ceil(Math.log10(Math.max(...rfs)) * 2) / 2;
    if (hi - lo < 1) {
      lo -= 0.5;
      hi += 0.5;
    }
    return { lo, hi };
  }, [tracks]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || width < 10) return;
    const ctx = setupCanvas(c, width, height);
    if (!ctx) return;
    const tk = chartTokens();
    ctx.fillStyle = tk.surface;
    ctx.fillRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2;
    const R = Math.min(width, height) / 2 - (compact ? 18 : 26);
    ctx.font = `10px ${tk.mono}`;
    ctx.lineWidth = 1;

    // rings (log frequency)
    const nRings = 4;
    for (let k = 1; k <= nRings; k++) {
      const r = (R * k) / nRings;
      ctx.strokeStyle = k === nRings ? tk.axis : tk.grid;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      const mhz = Math.pow(10, scale.lo + ((scale.hi - scale.lo) * k) / nRings);
      // ring labels sit just inside the ring, along the 0 deg spoke, so they
      // never collide with the compass labels drawn outside the outer ring
      ctx.fillStyle = tk.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(fmtFreqShort(mhz), cx + 4, cy - r + 2);
    }
    // spokes
    for (let d = 0; d < 360; d += 30) {
      const a = ((d - 90) * Math.PI) / 180;
      ctx.strokeStyle = d % 90 === 0 ? tk.axis : tk.grid;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a));
      ctx.stroke();
      if (d % 90 === 0 || !compact) {
        ctx.fillStyle = d % 90 === 0 ? tk.ink2 : tk.muted;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const lr = R + (compact ? 9 : 13);
        ctx.fillText(String(d).padStart(3, '0'), cx + lr * Math.cos(a), cy + lr * Math.sin(a));
      }
    }

    // marks
    const colour: Record<Cat, string> = { comint: tk.s1, resm: tk.s2, both: tk.s3 };
    const pos: { id: number; x: number; y: number; r: number }[] = [];
    for (const t of tracks) {
      if (!Number.isFinite(t.rf_mhz) || t.rf_mhz <= 0 || !Number.isFinite(t.aoa)) continue;
      const rr = ((Math.log10(t.rf_mhz) - scale.lo) / (scale.hi - scale.lo)) * R;
      const a = ((t.aoa - 90) * Math.PI) / 180;
      const x = cx + rr * Math.cos(a);
      const y = cy + rr * Math.sin(a);
      const r = 4 + Math.min(5, Math.log2(t.hits + 1));
      pos.push({ id: t.track_id, x, y, r });
      const sel = t.track_id === selectedId;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = colour[catOf(t)];
      ctx.globalAlpha = selectedId !== null && !sel ? 0.45 : 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = 2;
      ctx.strokeStyle = tk.surface; // 2px surface ring separates overlapping marks
      ctx.stroke();
      if (sel) {
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = tk.ink;
        ctx.beginPath();
        ctx.arc(x, y, r + 4, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = tk.ink;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(`T${t.track_id} ${fmtFreqShort(t.rf_mhz)} ${fmtDeg(t.aoa)}`, x + r + 7, y);
      }
      ctx.lineWidth = 1;
    }
    posRef.current = pos;
  }, [tracks, selectedId, width, height, compact, scale, themeV]);

  const nearest = (x: number, y: number) => {
    let best: { id: number; d: number } | null = null;
    for (const p of posRef.current) {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d <= Math.max(12, p.r + 6) && (!best || d < best.d)) best = { id: p.id, d };
    }
    return best?.id ?? null;
  };

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const x = e.nativeEvent.offsetX;
    const y = e.nativeEvent.offsetY;
    const id = nearest(x, y);
    if (id === null) return setHover(null);
    const t = tracks.find((k) => k.track_id === id);
    if (!t) return setHover(null);
    setHover({ x, y, text: `T${t.track_id}  ${fmtFreqShort(t.rf_mhz)}  AOA ${fmtDeg(t.aoa)}  ${t.modulation}  hits ${t.hits}` });
  };
  const onClick = (e: MouseEvent<HTMLCanvasElement>) => {
    if (!onSelect) return;
    const id = nearest(e.nativeEvent.offsetX, e.nativeEvent.offsetY);
    onSelect(id);
  };

  const counts = useMemo(() => {
    const c: Record<Cat, number> = { comint: 0, resm: 0, both: 0 };
    for (const t of tracks) c[catOf(t)]++;
    return c;
  }, [tracks]);

  return (
    <div>
      <div className="canvas-wrap" ref={wrapRef} style={{ height }}>
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`Bearing rose with ${tracks.length} tracks; angle is AOA, radius is log frequency`}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          onClick={onClick}
          style={{ cursor: onSelect ? 'pointer' : 'default' }}
        />
        {hover && (
          <div className="tooltip" style={{ left: Math.min(hover.x + 12, Math.max(0, width - 280)), top: Math.max(0, hover.y - 30) }}>
            {hover.text}
          </div>
        )}
      </div>
      <div className="legend" aria-label="Legend">
        <span>
          <i className="sw" style={{ background: 'var(--s1)' }} />
          comint only ({counts.comint})
        </span>
        <span>
          <i className="sw" style={{ background: 'var(--s2)' }} />
          resm only ({counts.resm})
        </span>
        <span>
          <i className="sw" style={{ background: 'var(--s3)' }} />
          fused comint+resm ({counts.both})
        </span>
        <span className="muted">radius = log RF, size = hits</span>
      </div>
    </div>
  );
}

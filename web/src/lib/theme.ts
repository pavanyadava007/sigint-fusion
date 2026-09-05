export type Theme = 'dark' | 'light';

export function readTheme(): Theme {
  try {
    const t = localStorage.getItem('sigint.theme');
    if (t === 'light' || t === 'dark') return t;
  } catch {
    /* storage unavailable */
  }
  return 'dark';
}

export function applyTheme(t: Theme): void {
  document.documentElement.setAttribute('data-theme', t);
  try {
    localStorage.setItem('sigint.theme', t);
  } catch {
    /* storage unavailable */
  }
}

/** Read the current CSS custom property values used by canvas charts. */
export function chartTokens() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name: string) => cs.getPropertyValue(name).trim();
  return {
    surface: v('--surface'),
    ink: v('--ink'),
    ink2: v('--ink-2'),
    muted: v('--muted'),
    grid: v('--grid'),
    axis: v('--axis'),
    s1: v('--s1'),
    s2: v('--s2'),
    s3: v('--s3'),
    s4: v('--s4'),
    s7: v('--s7'),
    critical: v('--critical'),
    mono: v('--mono') || 'monospace',
  };
}

/** Prepare a canvas for HiDPI drawing; returns the 2D context scaled to CSS px. */
export function setupCanvas(canvas: HTMLCanvasElement, cssW: number, cssH: number): CanvasRenderingContext2D | null {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(cssW * dpr));
  const h = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

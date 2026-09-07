// Small seedable PRNG (mulberry32) with a Box-Muller normal, so /synth is reproducible per seed.

export class Rng {
  private s: number;
  private spare: number | null = null;
  constructor(seed: number) {
    this.s = (seed >>> 0) || 0x9e3779b9;
  }
  uniform(lo = 0, hi = 1): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const u = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return lo + (hi - lo) * u;
  }
  int(n: number): number {
    return Math.floor(this.uniform() * n) % n;
  }
  normal(mean = 0, sd = 1): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return mean + sd * v;
    }
    let u = 0;
    while (u === 0) u = this.uniform();
    const v = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return mean + sd * r * Math.cos(2 * Math.PI * v);
  }
}

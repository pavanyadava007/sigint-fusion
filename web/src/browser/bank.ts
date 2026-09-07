// Demo assets for browser mode: the ONNX classifier, its class list, a bank of real RadioML frames,
// the knowledge base corpus + emitter catalogue and the model card. Everything is fetched once from
// ./demo/ relative to the page (works under a Space sub-path with base './').

import type { Rng } from './rng';

export interface FramesMeta {
  classes: string[];
  n: number;
  length: number;
  labels: number[];
  snr_db: number[];
  layout: string;
  source?: string;
}

export interface CorpusChunk {
  source: string;
  no: number;
  text: string;
}

export interface CatalogueEmitter {
  name: string;
  type: string;
  rf_min_mhz: number;
  rf_max_mhz: number;
  pri_us?: number | null;
  pw_us?: number | null;
  modulation: string;
  notes: string;
}

export interface Corpus {
  chunks: CorpusChunk[];
  emitters: CatalogueEmitter[];
}

export interface ModelCard {
  model: string;
  trained_on: string;
  val_top1: number;
  acc_vs_snr: Record<string, number>;
  device: string;
  n_train: number;
  classes: string[];
}

export interface Frame {
  i: Float32Array;
  q: Float32Array;
  snr_db: number;
  index: number;
}

/** Real RadioML frames grouped by modulation, [n][2][length] float32 (I row then Q row). */
export class FrameBank {
  readonly byMod = new Map<string, number[]>();
  constructor(
    readonly meta: FramesMeta,
    private readonly data: Float32Array,
  ) {
    meta.labels.forEach((lab, idx) => {
      const name = meta.classes[lab];
      if (!this.byMod.has(name)) this.byMod.set(name, []);
      this.byMod.get(name)!.push(idx);
    });
  }
  get modulations(): string[] {
    return this.meta.classes.filter((c) => this.byMod.has(c));
  }
  has(mod: string): boolean {
    return this.byMod.has(mod);
  }
  frame(index: number): Frame {
    const L = this.meta.length;
    const off = index * 2 * L;
    return {
      i: this.data.subarray(off, off + L),
      q: this.data.subarray(off + L, off + 2 * L),
      snr_db: this.meta.snr_db[index],
      index,
    };
  }
  draw(mod: string, rng: Rng): Frame {
    const idx = this.byMod.get(mod);
    if (!idx || !idx.length) throw new Error(`no frames for ${mod}`);
    return this.frame(idx[rng.int(idx.length)]);
  }
}

export interface Assets {
  classes: string[];
  bank: FrameBank;
  corpus: Corpus;
  card: ModelCard;
  modelBytes: Uint8Array;
}

/** The original fetch, captured before the shim wraps window.fetch. */
export const nativeFetch: typeof fetch = window.fetch.bind(window);

function assetUrl(name: string): string {
  return new URL(`demo/${name}`, document.baseURI).href;
}

async function get(name: string): Promise<Response> {
  const r = await nativeFetch(assetUrl(name), { credentials: 'same-origin' });
  if (!r.ok) throw new Error(`failed to load ${name}: HTTP ${r.status}`);
  return r;
}

export async function loadAssets(): Promise<Assets> {
  const [classes, meta, f32, corpus, card, model] = await Promise.all([
    get('classes.json').then((r) => r.json() as Promise<string[]>),
    get('frames.json').then((r) => r.json() as Promise<FramesMeta>),
    get('frames.f32').then((r) => r.arrayBuffer()),
    get('corpus.json').then((r) => r.json() as Promise<Corpus>),
    get('model_card.json').then((r) => r.json() as Promise<ModelCard>),
    get('model.onnx').then((r) => r.arrayBuffer()),
  ]);
  const expected = meta.n * 2 * meta.length;
  if (f32.byteLength !== expected * 4) throw new Error(`frames.f32 has ${f32.byteLength} bytes, expected ${expected * 4}`);
  const data = new Float32Array(f32); // little-endian on every platform the browser runs on
  return { classes, bank: new FrameBank(meta, data), corpus, card, modelBytes: new Uint8Array(model) };
}

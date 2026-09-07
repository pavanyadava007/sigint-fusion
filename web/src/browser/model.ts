// ONNX Runtime Web (wasm execution provider, single thread, main thread) wrapper around the
// ResNet-1D modulation classifier. Input "iq" float32 [batch, 2, length], output "logits" [batch, 21].

import * as ort from 'onnxruntime-web/wasm';
import { softmax } from './dsp';

export interface Classification {
  probs: number[];
  latencyMs: number;
}

/** Same normalisation as ml-service Model.predict: x / (sqrt(sum(x^2) / L) + 1e-8). */
export function rmsNormalise(i: ArrayLike<number>, q: ArrayLike<number>): Float32Array {
  const L = i.length;
  const out = new Float32Array(2 * L);
  let ss = 0;
  for (let k = 0; k < L; k++) ss += i[k] * i[k] + q[k] * q[k];
  const s = 1 / (Math.sqrt(ss / L) + 1e-8);
  for (let k = 0; k < L; k++) {
    out[k] = i[k] * s;
    out[L + k] = q[k] * s;
  }
  return out;
}

export class Classifier {
  private constructor(
    private readonly session: ort.InferenceSession,
    readonly classes: string[],
  ) {}

  static async load(modelBytes: Uint8Array, classes: string[]): Promise<Classifier> {
    // The wasm binary and its loader are copied next to index.html by the vite plugin; resolve
    // './' against the page so it works under a Space sub-path regardless of where the JS chunk lives.
    ort.env.wasm.wasmPaths = new URL('./', document.baseURI).href;
    ort.env.wasm.numThreads = 1; // no SharedArrayBuffer on static hosts; also avoids spawning workers
    ort.env.wasm.proxy = false;
    ort.env.logLevel = 'error';
    const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    return new Classifier(session, classes);
  }

  get inputName(): string {
    return this.session.inputNames[0] ?? 'iq';
  }

  async classify(i: ArrayLike<number>, q: ArrayLike<number>): Promise<Classification> {
    const L = i.length;
    const x = rmsNormalise(i, q);
    const tensor = new ort.Tensor('float32', x, [1, 2, L]);
    const t0 = performance.now();
    const out = await this.session.run({ [this.inputName]: tensor });
    const latencyMs = performance.now() - t0;
    const logits = out[this.session.outputNames[0] ?? 'logits'].data as Float32Array;
    return { probs: softmax(logits), latencyMs };
  }
}

// Entry point for browser mode (vite --mode browser): install the fetch shim synchronously so the
// first polls are answered in-page, then load the demo assets + ONNX model and start the scenario.

import { loadAssets } from './bank';
import { Classifier } from './model';
import { installFetchShim, Shim } from './shim';
import { Store } from './sim';

declare global {
  interface Window {
    __sigintDemo?: { store: Store; shim: Shim };
  }
}

let started: Promise<Shim> | null = null;

export function installBrowserMode(): Promise<Shim> {
  if (started) return started;
  const t0 = Date.now();
  started = (async () => {
    const assets = await loadAssets();
    const clf = await Classifier.load(assets.modelBytes, assets.classes);
    const store = new Store(assets, clf);
    const shim = new Shim(store);
    store.start();
    window.__sigintDemo = { store, shim };
    console.info(`browser demo ready: ${assets.bank.meta.n} real frames (${assets.bank.modulations.join(', ')}), ${clf.classes.length} classes, ${assets.corpus.chunks.length} corpus chunks, ${assets.corpus.emitters.length} catalogue emitters`);
    return shim;
  })();
  installFetchShim(started, t0);
  started.catch((e) => console.error('browser demo failed to start', e));
  return started;
}

/// <reference types="vite/client" />
import { readFileSync } from 'node:fs';
import type { ServerResponse } from 'node:http';
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, type Plugin, type ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';

// Dev-only proxies. In production nginx.conf performs the same prefix stripping.
// When a target is down, answer 502 like nginx does so the UI shows the same
// "unreachable" state in dev and prod.
const proxy = (target: string, prefix: string): ProxyOptions => ({
  target,
  changeOrigin: true,
  rewrite: (path: string) => path.replace(new RegExp(`^${prefix}`), ''),
  timeout: 300_000,
  proxyTimeout: 300_000,
  configure: (p) => {
    p.on('error', (_err, _req, res) => {
      const r = res as ServerResponse;
      if (!r.headersSent) r.writeHead(502, { 'content-type': 'application/json' });
      r.end(JSON.stringify({ error: `upstream ${target} unreachable` }));
    });
  },
});

// 'virtual:sigint-browser' (imported by src/main.tsx) is the browser-mode entry src/browser/boot.ts
// when building with --mode browser and an empty stub otherwise, so the server build never loads
// onnxruntime-web or the demo simulation.
const VIRTUAL_ID = 'virtual:sigint-browser';
const HERE = dirname(fileURLToPath(import.meta.url));
function browserEntry(browser: boolean): Plugin {
  return {
    name: 'sigint-browser-entry',
    resolveId(id) {
      if (id === VIRTUAL_ID) return browser ? resolve(HERE, 'src/browser/boot.ts') : `\0${VIRTUAL_ID}`;
      return null;
    },
    load(id) {
      if (id === `\0${VIRTUAL_ID}`) return 'export function installBrowserMode() {}';
      return null;
    },
  };
}

// Browser mode only: copy onnxruntime-web's wasm binary and its loader module to the outDir root,
// where src/browser/model.ts points ort.env.wasm.wasmPaths ('./' relative to the page).
const ORT_FILES = ['ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.mjs'];
function copyOrtWasm(): Plugin {
  const require = createRequire(import.meta.url);
  // the package blocks './package.json' in its exports map; its main entry lives in dist/
  const dist = dirname(require.resolve('onnxruntime-web'));
  return {
    name: 'sigint-copy-ort-wasm',
    apply: 'build',
    generateBundle(_options, bundle) {
      for (const f of ORT_FILES) this.emitFile({ type: 'asset', fileName: f, source: readFileSync(join(dist, f)) });
      // Vite also emits a hashed copy under assets/ for ORT's internal `new URL(wasm, import.meta.url)`;
      // it is never fetched once wasmPaths is overridden, so drop the 14 MB duplicate.
      for (const key of Object.keys(bundle)) if (/^assets\/ort-wasm-simd-threaded-[\w-]+\.wasm$/.test(key)) delete bundle[key];
    },
  };
}

export default defineConfig(({ mode }) => {
  const browser = mode === 'browser';
  return {
    // browser mode is served from a static host under an arbitrary sub-path (HashRouter)
    base: browser ? './' : '/',
    plugins: [react(), browserEntry(browser), ...(browser ? [copyOrtWasm()] : [])],
    server: {
      port: 5173,
      proxy: {
        '/api/ml': proxy('http://localhost:8000', '/api/ml'),
        '/api/agent': proxy('http://localhost:8001', '/api/agent'),
        '/api/gateway': proxy('http://localhost:8080', '/api/gateway'),
      },
    },
    build: {
      sourcemap: false,
      chunkSizeWarningLimit: 900,
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom', 'react-router-dom'],
            charts: ['recharts'],
            markdown: ['marked', 'dompurify'],
            ...(browser ? { ort: ['onnxruntime-web/wasm'] } : {}),
          },
        },
      },
    },
  };
});

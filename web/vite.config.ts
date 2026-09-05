import type { ServerResponse } from 'node:http';
import { defineConfig, type ProxyOptions } from 'vite';
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

export default defineConfig({
  plugins: [react()],
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
        },
      },
    },
  },
});

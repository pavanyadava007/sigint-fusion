/// <reference types="vite/client" />

// Resolved by vite.config.ts: src/browser/boot.ts in `--mode browser`, an empty stub otherwise.
declare module 'virtual:sigint-browser' {
  export function installBrowserMode(): unknown;
}

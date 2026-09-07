import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, HashRouter } from 'react-router-dom';
import { App } from './App';
import './styles.css';

// Browser mode (vite --mode browser): the backends are replaced by an in-page fetch shim and the
// router uses hashes so deep links work on a static host. Everything else is identical. The
// 'virtual:sigint-browser' module is src/browser/boot.ts in browser mode and an empty stub otherwise
// (see vite.config.ts), so the server build never bundles onnxruntime-web.
const BROWSER_MODE = import.meta.env.MODE === 'browser';

async function start() {
  if (BROWSER_MODE) {
    const { installBrowserMode } = await import('virtual:sigint-browser');
    installBrowserMode();
  }
  const Router = BROWSER_MODE ? HashRouter : BrowserRouter;
  ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
    <React.StrictMode>
      <Router>
        <App />
      </Router>
    </React.StrictMode>,
  );
}

void start();

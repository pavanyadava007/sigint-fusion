import { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useTimer } from '../hooks/usePoll';
import { applyTheme, readTheme, type Theme } from '../lib/theme';
import { chipLabel, useHealth, type ChipState } from '../hooks/useHealth';
import { IconChat, IconGrid, IconMark, IconMoon, IconRadar, IconServer, IconSun, IconWave } from './Icons';

const NAV = [
  { to: '/', label: 'Overview', icon: IconGrid, end: true },
  { to: '/emitters', label: 'Emitters', icon: IconRadar },
  { to: '/analyst', label: 'Analyst', icon: IconChat },
  { to: '/lab', label: 'Signal lab', icon: IconWave },
  { to: '/system', label: 'System', icon: IconServer },
];

function Chip({ name, short, state, title }: { name: string; short: string; state: ChipState; title?: string }) {
  return (
    <span className="chip" data-state={state} title={title ?? `${name}: ${chipLabel(state)}`} role="status">
      <span className="dot" aria-hidden />
      <span className="name">{name}</span>
      <span className="name-short" aria-hidden>
        {short}
      </span>
      <span className="state">{chipLabel(state)}</span>
    </span>
  );
}

function Clock() {
  const now = useTimer(1000);
  const utc = now.toISOString().slice(11, 19);
  return (
    <time className="clock" dateTime={now.toISOString()} title={now.toString()}>
      {now.toISOString().slice(0, 10)} {utc}Z
    </time>
  );
}

export function Layout() {
  const h = useHealth();
  const [theme, setTheme] = useState<Theme>(readTheme);
  useEffect(() => applyTheme(theme), [theme]);

  const llmTitle = h.agent.data?.llm
    ? `LLM ${h.agent.data.llm.model ?? ''} at ${h.agent.data.llm.base_url ?? ''}: ${chipLabel(h.llmState)}`
    : 'LLM status is reported by the agent; agent unreachable';

  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <IconMark className="brand-mark" />
          <div>
            <div className="brand-name">SIGINT-Fusion</div>
            <span className="brand-sub">operator console</span>
          </div>
        </div>
        <nav className="nav" aria-label="Primary">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end} title={n.label}>
              <n.icon />
              <span>{n.label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="rail-foot">
          <span>model: {h.ml.data?.model ?? '-'}</span>
          <span>ml v{h.ml.data?.version ?? '-'}</span>
        </div>
      </aside>
      <header className="topbar">
        <h1>
          <PageTitle />
        </h1>
        <div className="chips" aria-label="Backend health">
          <Chip name="gateway" short="gw" state={h.gatewayState} />
          <Chip name="ml-service" short="ml" state={h.mlState} />
          <Chip name="agent" short="ag" state={h.agentState} />
          <Chip name="llm" short="llm" state={h.llmState} title={llmTitle} />
        </div>
        <Clock />
        <button
          className="icon-btn"
          type="button"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Light theme' : 'Dark theme'}
        >
          {theme === 'dark' ? <IconSun /> : <IconMoon />}
        </button>
      </header>
      <main className="main" id="main">
        <Outlet />
      </main>
    </div>
  );
}

function PageTitle() {
  const { pathname } = useLocation();
  const n = NAV.find((x) => (x.end ? pathname === x.to : pathname.startsWith(x.to)));
  return <>{n?.label ?? 'Console'}</>;
}

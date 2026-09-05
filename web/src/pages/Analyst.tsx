import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import { agent, toApiError, type AskResult, type Report, type TraceStep } from '../api';
import { IconClose, IconDoc } from '../components/Icons';
import { Markdown } from '../components/Markdown';
import { EmptyState, Panel } from '../components/Panel';
import { usePoll, useTimer } from '../hooks/usePoll';
import { fmtDateTime, fmtNum, fmtTime } from '../lib/format';

const AGENT = 'agent';

interface Msg {
  id: number;
  role: 'user' | 'assistant' | 'error';
  text: string;
  at: number;
  trace?: TraceStep[];
  seconds?: number;
  model?: string;
}

const SUGGESTIONS = [
  'What is transmitting near 9400 MHz?',
  'Summarise the last 10 minutes of activity',
  'Which emitters are seen by more than one sensor?',
  'Any low-confidence detections I should re-check?',
  'Write an intelligence report on the current EOB',
];

function threadId(): string {
  const KEY = 'sigint.thread';
  try {
    const t = sessionStorage.getItem(KEY);
    if (t) return t;
    const n = `ui-${Math.random().toString(36).slice(2, 10)}`;
    sessionStorage.setItem(KEY, n);
    return n;
  } catch {
    return 'ui';
  }
}

function loadMsgs(thread: string): Msg[] {
  try {
    const raw = sessionStorage.getItem(`sigint.msgs.${thread}`);
    return raw ? (JSON.parse(raw) as Msg[]) : [];
  } catch {
    return [];
  }
}

function Trace({ trace }: { trace: TraceStep[] }) {
  const total = trace.reduce((a, s) => a + (s.seconds || 0), 0);
  return (
    <details className="trace">
      <summary>
        tool trace: {trace.length} call{trace.length === 1 ? '' : 's'}, {fmtNum(total, 2)} s in tools
      </summary>
      {trace.map((s, i) => (
        <div className="trace-step" key={i}>
          <span className="n">{i + 1}</span>
          <span className="tool">{s.tool}</span>
          <span className="dur">{fmtNum(s.seconds, 2)} s</span>
          <pre>{JSON.stringify(s.args ?? {}, null, 0)}</pre>
          {s.result_preview && <pre className="result">{s.result_preview}</pre>}
        </div>
      ))}
    </details>
  );
}

function Working({ since }: { since: number }) {
  const now = useTimer(500).getTime();
  return (
    <div className="working" role="status">
      <span className="pulse" aria-hidden />
      analyst working, {Math.max(0, (now - since) / 1000).toFixed(0)} s elapsed (tool calls and report writing can take a few minutes)
    </div>
  );
}

export function Analyst() {
  const [thread] = useState(threadId);
  const [msgs, setMsgs] = useState<Msg[]>(() => loadMsgs(thread));
  const [q, setQ] = useState('');
  const [busy, setBusy] = useState<number | null>(null);
  const [open, setOpen] = useState<Report | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const reports = usePoll(() => agent.reports(20), 15000);

  useEffect(() => {
    try {
      sessionStorage.setItem(`sigint.msgs.${thread}`, JSON.stringify(msgs.slice(-60)));
    } catch {
      /* ignore */
    }
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [msgs, thread, busy]);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKey = (e: globalThis.KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(null);
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open]);

  const send = async (question: string) => {
    const text = question.trim();
    if (!text || busy !== null) return;
    setQ('');
    const at = Date.now();
    setMsgs((m) => [...m, { id: at, role: 'user', text, at }]);
    setBusy(at);
    try {
      const r: AskResult = await agent.ask(text, thread);
      setMsgs((m) => [
        ...m,
        { id: Date.now(), role: 'assistant', text: r.answer ?? '', at: Date.now(), trace: r.trace, seconds: r.seconds, model: r.model },
      ]);
      reports.refresh();
    } catch (e) {
      const err = toApiError(e);
      setMsgs((m) => [
        ...m,
        {
          id: Date.now(),
          role: 'error',
          at: Date.now(),
          text:
            err.kind === 'timeout'
              ? `agent did not answer within 300 s (${err.message})`
              : `agent unreachable or failed: ${err.message}`,
        },
      ]);
    } finally {
      setBusy(null);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void send(q);
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void send(q);
    }
  };

  return (
    <div className="page">
      <div className="grid analyst-grid">
        <Panel
          title="Analyst"
          flush
          tools={
            <>
              <span>thread {thread}</span>
              <button
                type="button"
                className="btn"
                style={{ padding: '2px 8px', fontSize: 11 }}
                onClick={() => setMsgs([])}
                disabled={!msgs.length || busy !== null}
              >
                clear
              </button>
            </>
          }
        >
          <div className="chat">
            <div className="chat-log" ref={logRef} aria-live="polite">
              {msgs.length === 0 && (
                <div className="empty" style={{ flexDirection: 'column', gap: 4 }}>
                  <span>Ask the analyst about the current picture. Answers are grounded in the emitter database, the doctrine corpus and live tools.</span>
                  <span>Use the prompts below to start.</span>
                </div>
              )}
              {msgs.map((m) => (
                <div className={`msg ${m.role}`} key={m.id}>
                  <div className="msg-meta">
                    <span>{m.role === 'user' ? 'operator' : m.role === 'assistant' ? 'analyst' : 'error'}</span>
                    <span>{fmtTime(new Date(m.at).toISOString())}Z</span>
                    {m.role === 'assistant' && m.seconds !== undefined && <span>{fmtNum(m.seconds, 1)} s</span>}
                    {m.role === 'assistant' && m.model && <span>{m.model}</span>}
                  </div>
                  {m.role === 'assistant' ? (
                    <>
                      <Markdown text={m.text || '(empty answer)'} />
                      {m.trace && m.trace.length > 0 && <Trace trace={m.trace} />}
                    </>
                  ) : (
                    <div className="bubble">{m.text}</div>
                  )}
                </div>
              ))}
              {busy !== null && <Working since={busy} />}
            </div>
            <div className="chips-row" aria-label="Suggested prompts">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" onClick={() => void send(s)} disabled={busy !== null}>
                  {s}
                </button>
              ))}
            </div>
            <form className="chat-form" onSubmit={onSubmit}>
              <textarea
                aria-label="Question for the analyst"
                placeholder="Ask the analyst (Enter to send, Shift+Enter for newline)"
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={onKey}
                rows={2}
              />
              <button className="btn primary" type="submit" disabled={busy !== null || !q.trim()}>
                {busy !== null ? 'Working' : 'Ask'}
              </button>
            </form>
          </div>
        </Panel>

        <Panel
          title="Reports"
          flush
          service={AGENT}
          error={reports.error}
          updatedAt={reports.updatedAt}
          stale={!!reports.error && !!reports.data?.length}
          tools={<span>{reports.data?.length ?? 0}</span>}
        >
          {!reports.data?.length ? (
            <EmptyState service={AGENT} error={reports.error} loading={reports.loading} message="no saved reports yet" height={120} />
          ) : (
            <ul className="reports">
              {reports.data.map((r) => (
                <li key={r.id}>
                  <button type="button" onClick={() => setOpen(r)}>
                    <span className="q">
                      <IconDoc style={{ width: 12, height: 12, verticalAlign: -2, marginRight: 6 }} />
                      {r.question || `report #${r.id}`}
                    </span>
                    <span className="t">
                      #{r.id} {fmtDateTime(r.ts)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      {open && (
        <>
          <div className="drawer-backdrop" onClick={() => setOpen(null)} aria-hidden />
          <aside className="drawer" role="dialog" aria-modal="true" aria-label={`Report ${open.id}`}>
            <div className="drawer-head">
              <h3 title={open.question}>
                #{open.id} {open.question}
              </h3>
              <span className="muted mono" style={{ fontSize: 11 }}>
                {fmtDateTime(open.ts)}
              </span>
              <button ref={closeRef} className="icon-btn" type="button" onClick={() => setOpen(null)} aria-label="Close report">
                <IconClose />
              </button>
            </div>
            <div className="drawer-body">
              <Markdown text={open.report || '(empty report)'} />
            </div>
          </aside>
        </>
      )}
    </div>
  );
}

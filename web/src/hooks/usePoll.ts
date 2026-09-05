import { useCallback, useEffect, useRef, useState, type DependencyList } from 'react';
import { ApiError, toApiError } from '../api';

export interface PollState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
  updatedAt: number | null;
  refresh: () => void;
}

/**
 * Poll an async function on a fixed interval. The next call is scheduled only
 * after the previous one settles (no overlap). Polling pauses while the tab is
 * hidden and resumes with an immediate fetch when it becomes visible. On error
 * the last good data is retained so panels can hold their previous render.
 */
export function usePoll<T>(
  fn: () => Promise<T>,
  intervalMs: number,
  deps: DependencyList = [],
  enabled = true,
): PollState<T> {
  const [state, setState] = useState<Omit<PollState<T>, 'refresh'>>({
    data: null,
    error: null,
    loading: true,
    updatedAt: null,
  });
  const [tick, setTick] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let inFlight = false;
    let timer: number | undefined;

    const schedule = () => {
      if (!cancelled) timer = window.setTimeout(run, intervalMs);
    };
    const run = async () => {
      if (cancelled || inFlight) return;
      if (document.hidden) {
        schedule();
        return;
      }
      inFlight = true;
      try {
        const data = await fnRef.current();
        if (!cancelled) setState({ data, error: null, loading: false, updatedAt: Date.now() });
      } catch (e) {
        if (!cancelled) setState((s) => ({ ...s, error: toApiError(e), loading: false }));
      } finally {
        inFlight = false;
        schedule();
      }
    };
    const onVisibility = () => {
      if (!document.hidden) {
        window.clearTimeout(timer);
        void run();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    void run();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, enabled, tick, ...deps]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, refresh };
}

/** Re-render on a fixed interval (paused while hidden). Returns the current time. */
export function useTimer(intervalMs = 1000): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    let timer: number | undefined;
    const tickFn = () => {
      if (!document.hidden) setNow(new Date());
      timer = window.setTimeout(tickFn, intervalMs);
    };
    timer = window.setTimeout(tickFn, intervalMs);
    const onVisibility = () => {
      if (!document.hidden) setNow(new Date());
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [intervalMs]);
  return now;
}

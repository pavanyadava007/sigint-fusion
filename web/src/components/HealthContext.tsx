import type { ReactNode } from 'react';
import { agent, gateway, ml } from '../api';
import { HealthCtx, type ChipState, type HealthSnapshot } from '../hooks/useHealth';
import { usePoll, type PollState } from '../hooks/usePoll';

function stateOf<T>(p: PollState<T>, judge: (d: T) => ChipState): ChipState {
  if (p.error) return 'down';
  if (!p.data) return 'unknown';
  return judge(p.data);
}

/** Polls the three backend health endpoints once for the whole app. */
export function HealthProvider({ children }: { children: ReactNode }) {
  const mlH = usePoll(ml.health, 5000);
  const agH = usePoll(agent.health, 10000);
  const gwH = usePoll(gateway.health, 10000);

  const llm = agH.data?.llm;
  const value: HealthSnapshot = {
    ml: mlH,
    agent: agH,
    gateway: gwH,
    mlState: stateOf(mlH, (d) => (d.status === 'ok' && d.model_loaded ? 'ok' : 'degraded')),
    agentState: stateOf(agH, (d) => (d.status === 'ok' ? 'ok' : 'degraded')),
    gatewayState: stateOf(gwH, (d) => (d.status === 'UP' ? 'ok' : d.status === 'UNKNOWN' ? 'unknown' : 'degraded')),
    llmState: agH.error || !agH.data ? 'unknown' : llm?.reachable === true ? 'ok' : llm?.reachable === false ? 'down' : 'unknown',
  };
  return <HealthCtx.Provider value={value}>{children}</HealthCtx.Provider>;
}

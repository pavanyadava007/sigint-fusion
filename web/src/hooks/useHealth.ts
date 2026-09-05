import { createContext, useContext } from 'react';
import type { AgentHealth, GatewayHealth, MlHealth } from '../api';
import type { PollState } from './usePoll';

export type ChipState = 'ok' | 'degraded' | 'down' | 'unknown';

export interface HealthSnapshot {
  ml: PollState<MlHealth>;
  agent: PollState<AgentHealth>;
  gateway: PollState<GatewayHealth>;
  mlState: ChipState;
  agentState: ChipState;
  gatewayState: ChipState;
  llmState: ChipState;
}

export const HealthCtx = createContext<HealthSnapshot | null>(null);

export function useHealth(): HealthSnapshot {
  const v = useContext(HealthCtx);
  if (!v) throw new Error('useHealth outside HealthProvider');
  return v;
}

export function chipLabel(s: ChipState): string {
  return s === 'ok' ? 'up' : s === 'degraded' ? 'degraded' : s === 'down' ? 'down' : 'unknown';
}

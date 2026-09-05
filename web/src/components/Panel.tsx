import type { ReactNode } from 'react';
import { ApiError, describeError } from '../api';
import { IconOffline, IconWarn } from './Icons';

interface PanelProps {
  title: string;
  tools?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  className?: string;
  style?: React.CSSProperties;
  /** Backend the panel depends on; when `error` is set a status line appears. */
  service?: string;
  error?: ApiError | null;
  updatedAt?: number | null;
  stale?: boolean;
}

export function Panel({ title, tools, children, flush, className, style, service, error, updatedAt, stale }: PanelProps) {
  return (
    <section className={`panel ${className ?? ''}`} style={style} aria-label={title}>
      <header className="panel-head">
        <h3>{title}</h3>
        {tools && <div className="tools">{tools}</div>}
      </header>
      <div className={`panel-body ${flush ? 'flush' : ''} ${stale ? 'stale' : ''}`}>{children}</div>
      {service && error && (
        <div className="status-line" data-level="critical" role="status">
          <IconOffline />
          <span>{describeError(service, error)}</span>
          {updatedAt && <span className="muted">holding data from {new Date(updatedAt).toISOString().slice(11, 19)}Z</span>}
        </div>
      )}
    </section>
  );
}

/** Honest empty / unreachable state. Never renders placeholder data. */
export function EmptyState({
  service,
  error,
  message,
  loading,
  height,
}: {
  service?: string;
  error?: ApiError | null;
  message?: string;
  loading?: boolean;
  height?: number;
}) {
  const style = height ? { minHeight: height } : undefined;
  if (error && service) {
    return (
      <div className="empty" data-level="critical" style={style} role="status">
        <IconOffline />
        <span>{describeError(service, error)}</span>
      </div>
    );
  }
  if (loading) {
    return (
      <div className="empty" style={style} role="status">
        <span>waiting for {service ?? 'data'}...</span>
      </div>
    );
  }
  return (
    <div className="empty" style={style} role="status">
      <IconWarn />
      <span>{message ?? 'no data'}</span>
    </div>
  );
}

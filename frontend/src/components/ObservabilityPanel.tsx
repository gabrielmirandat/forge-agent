/**Observability panel component for real-time system metrics.

Displays:
- Global metrics: CPU, memory, disk, network, GPU (system-wide)
- Global LLM metrics: models used, total tokens, total calls, active sessions

Uses SSE for real-time updates.
*/

import { useEffect, useRef, useState } from 'react';

interface SystemMetrics {
  available: boolean;
  usage_percent?: number;
  cores?: number;
  frequency_mhz?: number;
  temperature_celsius?: number;
  total_gb?: number;
  used_gb?: number;
  available_gb?: number;
  error?: string;
}

interface ObservabilityData {
  timestamp: number;
  global: {
    system: {
      cpu: SystemMetrics;
      memory: SystemMetrics;
      disk: SystemMetrics;
      network: {
        available: boolean;
        bytes_sent_mb?: number;
        bytes_recv_mb?: number;
        error?: string;
      };
      gpu?: {
        available: boolean;
        name?: string;
        temperature_celsius?: number;
        utilization_percent?: number;
        memory_used_mb?: number;
        memory_total_mb?: number;
        power_draw_watts?: number;
        power_limit_watts?: number;
        error?: string;
      };
    };
    llm: {
      total_tokens: number;
      total_calls: number;
      models_used: string[];
      active_sessions: number;
      last_used_at?: number | null;
      model_avg_response_times?: Record<string, number>;
      per_model?: Record<string, {
        total_tokens: number;
        total_calls: number;
        active_sessions: number;
        avg_response_time?: number | null;
        avg_tokens_per_second?: number | null;
        last_tokens_per_second?: number | null;
        last_used_at?: number | null;
      }>;
    };
  };
}

export function ObservabilityPanel() {
  const [metrics, setMetrics] = useState<ObservabilityData | null>(null);
  const [connected, setConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isConnectingRef = useRef(false);
  const maxReconnectAttempts = 10; // Maximum reconnect attempts before giving up
  const baseReconnectDelay = 2000; // Start with 2 second delay

  useEffect(() => {
    const connect = () => {
      // Prevent multiple simultaneous connection attempts
      if (isConnectingRef.current || (eventSourceRef.current && eventSourceRef.current.readyState === EventSource.CONNECTING)) {
        return;
      }

      // Close existing connection if any
      if (eventSourceRef.current) {
        try {
          eventSourceRef.current.close();
        } catch (e) {
          // Ignore errors when closing
        }
        eventSourceRef.current = null;
      }

      isConnectingRef.current = true;

      // Get SSE URL
      // Use relative path - Vite proxy will handle it
      const sseUrl = '/api/v1/observability/metrics';

      try {
        const eventSource = new EventSource(sseUrl);
        eventSourceRef.current = eventSource;

        eventSource.onopen = () => {
          console.log('Observability SSE connected');
          setConnected(true);
          isConnectingRef.current = false;
          reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
          
          // Clear any pending reconnect
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
          }
        };

        eventSource.onmessage = (event) => {
          try {
            const parsed = JSON.parse(event.data);
            
            // Handle different event types
            if (parsed.type === 'metrics' && parsed.data) {
              setMetrics(parsed.data as ObservabilityData);
            } else if (parsed.type === 'connection') {
              console.log('Observability SSE connection established');
            } else if (parsed.type === 'heartbeat') {
              // Heartbeat received, connection is alive
              setConnected(true);
            } else if (parsed.type === 'error') {
              console.error('Observability SSE error:', parsed.data);
            }
          } catch (e) {
            console.error('Failed to parse metrics:', e);
          }
        };

        eventSource.onerror = (error) => {
          console.error('SSE error:', error);
          setConnected(false);
          isConnectingRef.current = false;
          
          // EventSource automatically reconnects, but we track attempts
          if (eventSource.readyState === EventSource.CLOSED) {
            // Connection closed - try to reconnect manually
            if (reconnectAttemptsRef.current < maxReconnectAttempts) {
              const delay = Math.min(
                baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current),
                10000 // Max 10 seconds
              );
              
              reconnectAttemptsRef.current += 1;
              console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);

              reconnectTimeoutRef.current = window.setTimeout(() => {
                reconnectTimeoutRef.current = null;
                connect();
              }, delay);
            } else {
              console.warn('Max reconnect attempts reached. Stopping reconnection attempts.');
              // Reset after a longer delay (30 seconds) to allow backend to restart
              reconnectTimeoutRef.current = window.setTimeout(() => {
                reconnectAttemptsRef.current = 0;
                reconnectTimeoutRef.current = null;
                connect();
              }, 30000);
            }
          }
        };
      } catch (error) {
        console.error('Failed to create EventSource:', error);
        setConnected(false);
        isConnectingRef.current = false;
        
        // Retry with exponential backoff
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          const delay = Math.min(
            baseReconnectDelay * Math.pow(2, reconnectAttemptsRef.current),
            10000
          );
          
          reconnectAttemptsRef.current += 1;
          reconnectTimeoutRef.current = window.setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connect();
          }, delay);
        }
      }
    };

    connect();

    return () => {
      isConnectingRef.current = false;
      if (eventSourceRef.current) {
        try {
          eventSourceRef.current.close();
        } catch (e) {
          // Ignore errors
        }
        eventSourceRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, []);

  const getStatusColor = (percent?: number): string => {
    if (percent === undefined) return '#666';
    if (percent < 50) return '#4ade80'; // green
    if (percent < 80) return '#fbbf24'; // yellow
    return '#f87171'; // red
  };

  if (!metrics) {
    return (
      <div
        style={{
          width: '300px',
          padding: '1rem',
          background: '#1a1a1a',
          borderLeft: '1px solid #333',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#888',
        }}
      >
        <div style={{ marginBottom: '0.5rem' }}>
          {connected ? 'Loading metrics...' : 'Connecting...'}
        </div>
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: connected ? '#4ade80' : '#f87171',
          }}
        />
      </div>
    );
  }

  const globalLLM = metrics.global.llm;
  const system = metrics.global.system;

  return (
    <div
      style={{
        width: '300px',
        padding: '1rem',
        background: '#1a1a1a',
        borderLeft: '1px solid #333',
        overflowY: 'auto',
        fontSize: '0.875rem',
        color: '#e5e5e5',
      }}
    >
      {/* Connection status */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '1rem',
          paddingBottom: '0.75rem',
          borderBottom: '1px solid #333',
        }}
      >
        <div style={{ fontWeight: '600', fontSize: '0.9rem' }}>Observability</div>
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: connected ? '#4ade80' : '#f87171',
          }}
          title={connected ? 'Connected' : 'Disconnected'}
        />
      </div>

      {/* GLOBAL METRICS SECTION */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div
          style={{
            fontSize: '0.75rem',
            fontWeight: '600',
            color: '#888',
            textTransform: 'uppercase',
            marginBottom: '0.75rem',
            letterSpacing: '0.5px',
          }}
        >
          🌐 Global
        </div>

        {/* CPU */}
        {system.cpu.available && (
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ marginBottom: '0.5rem', fontWeight: '500' }}>CPU</div>
            {system.cpu.usage_percent !== undefined && (
              <div style={{ marginBottom: '0.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span>Usage</span>
                  <span style={{ color: getStatusColor(system.cpu.usage_percent) }}>
                    {system.cpu.usage_percent.toFixed(1)}%
                  </span>
                </div>
                <div
                  style={{
                    width: '100%',
                    height: '6px',
                    background: '#333',
                    borderRadius: '3px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${system.cpu.usage_percent}%`,
                      height: '100%',
                      background: getStatusColor(system.cpu.usage_percent),
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              </div>
            )}
            {system.cpu.temperature_celsius !== undefined && (
              <div style={{ fontSize: '0.8rem', color: '#aaa' }}>
                Temp: {system.cpu.temperature_celsius.toFixed(1)}°C
              </div>
            )}
          </div>
        )}

        {/* Memory */}
        {system.memory.available && (
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ marginBottom: '0.5rem', fontWeight: '500' }}>Memory</div>
            {system.memory.usage_percent !== undefined && (
              <div style={{ marginBottom: '0.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span>Usage</span>
                  <span style={{ color: getStatusColor(system.memory.usage_percent) }}>
                    {system.memory.usage_percent.toFixed(1)}%
                  </span>
                </div>
                <div
                  style={{
                    width: '100%',
                    height: '6px',
                    background: '#333',
                    borderRadius: '3px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${system.memory.usage_percent}%`,
                      height: '100%',
                      background: getStatusColor(system.memory.usage_percent),
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              </div>
            )}
            {system.memory.used_gb !== undefined && system.memory.total_gb !== undefined && (
              <div style={{ fontSize: '0.8rem', color: '#aaa' }}>
                {system.memory.used_gb.toFixed(1)} GB / {system.memory.total_gb.toFixed(1)} GB
              </div>
            )}
          </div>
        )}

        {/* GPU */}
        {system.gpu && (system.gpu.available || system.gpu.name || system.gpu.utilization_percent !== undefined) && (
          <div style={{ marginBottom: '1rem' }}>
            <div style={{ marginBottom: '0.5rem', fontWeight: '500' }}>GPU</div>
            {system.gpu.name && (
              <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '0.5rem' }}>
                {system.gpu.name}
              </div>
            )}
            {system.gpu.utilization_percent !== undefined && (
              <div style={{ marginBottom: '0.25rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                  <span>Usage</span>
                  <span style={{ color: getStatusColor(system.gpu.utilization_percent) }}>
                    {system.gpu.utilization_percent.toFixed(1)}%
                  </span>
                </div>
                <div
                  style={{
                    width: '100%',
                    height: '6px',
                    background: '#333',
                    borderRadius: '3px',
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${system.gpu.utilization_percent}%`,
                      height: '100%',
                      background: getStatusColor(system.gpu.utilization_percent),
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              </div>
            )}
            {system.gpu.memory_used_mb !== undefined && system.gpu.memory_total_mb !== undefined && (
              <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '0.25rem' }}>
                Memory: {(system.gpu.memory_used_mb / 1024).toFixed(1)} GB / {(system.gpu.memory_total_mb / 1024).toFixed(1)} GB
              </div>
            )}
            {system.gpu.temperature_celsius !== undefined && (
              <div style={{ fontSize: '0.8rem', color: '#aaa' }}>
                Temp: {system.gpu.temperature_celsius.toFixed(1)}°C
              </div>
            )}
            {system.gpu.power_draw_watts !== undefined && (
              <div style={{ fontSize: '0.8rem', color: '#aaa' }}>
                Power: {system.gpu.power_draw_watts.toFixed(1)}W
                {system.gpu.power_limit_watts !== undefined &&
                  ` / ${system.gpu.power_limit_watts.toFixed(1)}W`}
              </div>
            )}
            {system.gpu.error && (
              <div style={{ fontSize: '0.75rem', color: '#f87171' }}>
                Error: {system.gpu.error}
              </div>
            )}
            {!system.gpu.available && !system.gpu.name && system.gpu.utilization_percent === undefined && (
              <div style={{ fontSize: '0.8rem', color: '#888' }}>
                GPU not available
              </div>
            )}
          </div>
        )}

        {/* Global LLM Metrics */}
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ marginBottom: '0.5rem', fontWeight: '500' }}>LLM</div>
          {globalLLM.per_model && Object.keys(globalLLM.per_model).length > 0 ? (
            <div style={{ fontSize: '0.75rem', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #333' }}>
                    <th style={{ textAlign: 'left', padding: '0.25rem 0.5rem', color: '#888', fontWeight: '600' }}>Model</th>
                    <th style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#888', fontWeight: '600' }}>Tokens</th>
                    <th style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#888', fontWeight: '600' }}>Calls</th>
                    <th style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#19c37d', fontWeight: '600' }}>tok/s</th>
                    <th style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#888', fontWeight: '600' }}>Last Used</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(globalLLM.per_model).map(([model, metrics]) => {
                    const formatLastUsed = (timestamp?: number | null): string => {
                      if (!timestamp) return 'Never';
                      const now = Date.now() / 1000;
                      const diff = now - timestamp;
                      if (diff < 60) return `${Math.floor(diff)}s ago`;
                      if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
                      if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
                      return `${Math.floor(diff / 86400)}d ago`;
                    };
                    const tps = metrics.last_tokens_per_second ?? metrics.avg_tokens_per_second;
                    return (
                      <tr key={model} style={{ borderBottom: '1px solid #2a2a2a' }}>
                        <td style={{ padding: '0.25rem 0.5rem', color: '#aaa', wordBreak: 'break-word' }}>{model}</td>
                        <td style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#aaa' }}>
                          {metrics.total_tokens.toLocaleString()}
                        </td>
                        <td style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#aaa' }}>
                          {metrics.total_calls}
                        </td>
                        <td style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: tps ? '#19c37d' : '#555', fontWeight: tps ? 600 : 400 }}>
                          {tps != null ? `${tps}` : '—'}
                        </td>
                        <td style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#aaa', fontSize: '0.7rem' }}>
                          {formatLastUsed(metrics.last_used_at)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ fontSize: '0.75rem', overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid #333' }}>
                    <th style={{ textAlign: 'left', padding: '0.25rem 0.5rem', color: '#888', fontWeight: '600' }}>Model</th>
                    <th style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#888', fontWeight: '600' }}>Tokens</th>
                    <th style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#888', fontWeight: '600' }}>Calls</th>
                    <th style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#19c37d', fontWeight: '600' }}>tok/s</th>
                    <th style={{ textAlign: 'right', padding: '0.25rem 0.5rem', color: '#888', fontWeight: '600' }}>Last Used</th>
                  </tr>
                </thead>
                <tbody>
                  {/* No data rows - table shows only headers */}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

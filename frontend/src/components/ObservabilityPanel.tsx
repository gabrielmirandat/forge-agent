/**Observability panel component for real-time system metrics.

Displays:
- Global metrics: CPU, memory, disk, network, GPU (system-wide)
- Global LLM metrics: models used, total tokens, total calls, active sessions

Uses WebSocket for real-time updates.
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
    };
  };
}

export function ObservabilityPanel() {
  const [metrics, setMetrics] = useState<ObservabilityData | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const isConnectingRef = useRef(false);
  const maxReconnectAttempts = 10; // Maximum reconnect attempts before giving up
  const baseReconnectDelay = 1000; // Start with 1 second delay

  useEffect(() => {
    const connect = () => {
      // Prevent multiple simultaneous connection attempts
      if (isConnectingRef.current || (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING)) {
        return;
      }

      // Close existing connection if any
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch (e) {
          // Ignore errors when closing
        }
        wsRef.current = null;
      }

      isConnectingRef.current = true;

      // Get WebSocket URL
      // In development, Vite proxy should handle /api routes (configured in vite.config.ts)
      // The proxy forwards /api/* to http://localhost:8000
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      
      // Use relative path - Vite proxy will forward WebSocket connections to backend
      // Make sure vite.config.ts has ws: true in the proxy configuration
      const wsUrl = `${protocol}//${host}/api/v1/ws/observability`;

      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          console.log('Observability WebSocket connected');
          setConnected(true);
          isConnectingRef.current = false;
          reconnectAttemptsRef.current = 0; // Reset reconnect attempts on successful connection
          
          // Clear any pending reconnect
          if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
          }
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            setMetrics(data as ObservabilityData);
          } catch (e) {
            console.error('Failed to parse metrics:', e);
          }
        };

        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setConnected(false);
          isConnectingRef.current = false;
        };

        ws.onclose = (event) => {
          console.log('Observability WebSocket disconnected', {
            code: event.code,
            reason: event.reason,
            wasClean: event.wasClean,
          });
          setConnected(false);
          isConnectingRef.current = false;
          wsRef.current = null;

          // Don't reconnect if it was a clean close (e.g., component unmounting)
          if (event.wasClean && event.code === 1000) {
            return;
          }

          // Exponential backoff: 1s, 2s, 4s, 8s, ... up to 10s max
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
        };
      } catch (error) {
        console.error('Failed to create WebSocket:', error);
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
      if (wsRef.current) {
        try {
          wsRef.current.close(1000, 'Component unmounting'); // Clean close
        } catch (e) {
          // Ignore errors
        }
        wsRef.current = null;
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
        {system.gpu?.available && (
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
          </div>
        )}

        {/* Global LLM Metrics */}
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ marginBottom: '0.5rem', fontWeight: '500' }}>LLM</div>
          {globalLLM.models_used.length > 0 && (
            <div style={{ fontSize: '0.8rem', color: '#aaa', marginBottom: '0.25rem' }}>
              Models: {globalLLM.models_used.join(', ')}
            </div>
          )}
          <div style={{ fontSize: '0.8rem', color: '#aaa' }}>
            Total Tokens: {globalLLM.total_tokens.toLocaleString()}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#aaa' }}>
            Total Calls: {globalLLM.total_calls}
          </div>
          <div style={{ fontSize: '0.8rem', color: '#aaa' }}>
            Active Sessions: {globalLLM.active_sessions}
          </div>
        </div>
      </div>
    </div>
  );
}

/**SSE hook for real-time event streaming.

Uses Server-Sent Events (SSE) for streaming agent reasoning and execution updates.
SSE is simpler and more reliable for one-way event streaming from server to client.
*/

import { useEffect, useRef, useState } from 'react';

export interface Event {
  type: string;
  properties: Record<string, any>;
  timestamp: number;
}

export type EventHandler = (event: Event) => void;

export function useEventStream(
  onEvent: EventHandler,
  enabled: boolean = true,
  sessionId: string | null | undefined = null
): { connected: boolean; error: string | null } {
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 2000; // 2 seconds

  // Stabilize onEvent callback to prevent infinite reconnections
  const stableOnEvent = useRef(onEvent);
  useEffect(() => {
    stableOnEvent.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    // session_id is now REQUIRED - don't connect without it
    if (!enabled || !sessionId) {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      setConnected(false);
      if (!sessionId && enabled) {
        setError("Session ID is required for event stream");
      } else {
        setError(null);
      }
      return;
    }

    const connect = () => {
      // Close existing connection if any
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }

      // Determine SSE URL with session_id filter (REQUIRED)
      // Use relative path - Vite proxy will handle it
      const sseUrl = `/api/v1/events/event?session_id=${encodeURIComponent(sessionId)}`;

      console.log('Connecting to SSE:', sseUrl);

      try {
        const eventSource = new EventSource(sseUrl);
        eventSourceRef.current = eventSource;

        eventSource.onopen = () => {
          console.log('SSE connection opened');
          setConnected(true);
          setError(null);
          reconnectAttempts.current = 0;
        };

        eventSource.onmessage = (e) => {
          try {
            const event: Event = JSON.parse(e.data);
            console.log('SSE event received:', event);
            // Use stable ref to avoid dependency issues
            stableOnEvent.current(event);
          } catch (err) {
            console.error('Error parsing SSE event:', err);
          }
        };

        eventSource.onerror = (err) => {
          console.error('SSE connection error:', err);
          setConnected(false);
          
          // EventSource automatically reconnects, but we track attempts
          if (eventSource.readyState === EventSource.CLOSED) {
            // Connection closed - try to reconnect manually
            if (reconnectAttempts.current < maxReconnectAttempts) {
              reconnectAttempts.current++;
              setError(`Reconnecting... (${reconnectAttempts.current}/${maxReconnectAttempts})`);
              
              reconnectTimeoutRef.current = window.setTimeout(() => {
                connect();
              }, reconnectDelay);
            } else {
              setError('Failed to connect to event stream');
            }
          }
        };
      } catch (err) {
        console.error('Error creating EventSource:', err);
        setError(err instanceof Error ? err.message : 'Failed to connect');
      }
    };

    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      setConnected(false);
    };
  }, [enabled, sessionId]); // Reconnect when sessionId changes

  return { connected, error };
}

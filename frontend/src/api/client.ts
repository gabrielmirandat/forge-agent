/**Thin API client - no retries, no transformation, no business logic.

Errors bubble up. No retries. No transformation.
API is the single source of truth.
*/

import type {
  ExecuteRequest,
  ExecuteResponse,
  PlanRequest,
  PlanResponse,
} from '../types/api';

const API_BASE = '/api/v1';

// Generate request ID for correlation
function generateRequestId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

async function fetchJson<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const requestId = generateRequestId();
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      'X-Request-Id': requestId,
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      error: `HTTP ${response.status}: ${response.statusText}`,
    }));
    throw new Error(error.error || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function plan(
  goal: string,
  context?: Record<string, unknown>
): Promise<PlanResponse> {
  const request: PlanRequest = {
    goal,
    ...(context && { context }),
  };

  return fetchJson<PlanResponse>(`${API_BASE}/plan`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function execute(
  plan: import('../types/api').Plan,
  executionPolicy?: {
    max_retries_per_step?: number;
    retry_delay_seconds?: number;
    rollback_on_failure?: boolean;
  }
): Promise<ExecuteResponse> {
  const request: ExecuteRequest = {
    plan,
    ...(executionPolicy && {
      execution_policy: {
        max_retries_per_step: executionPolicy.max_retries_per_step ?? 0,
        retry_delay_seconds: executionPolicy.retry_delay_seconds ?? 0.0,
        rollback_on_failure: executionPolicy.rollback_on_failure ?? false,
      },
    }),
  };

  return fetchJson<ExecuteResponse>(`${API_BASE}/execute`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

// Session/chat API
export async function createSession(
  title?: string
): Promise<import('../types/api').CreateSessionResponse> {
  const request: import('../types/api').CreateSessionRequest = {};
  if (title) {
    request.title = title;
  }
  return fetchJson<import('../types/api').CreateSessionResponse>(`${API_BASE}/sessions`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function listSessions(
  limit: number = 20,
  offset: number = 0
): Promise<import('../types/api').SessionsListResponse> {
  return fetchJson<import('../types/api').SessionsListResponse>(
    `${API_BASE}/sessions?limit=${limit}&offset=${offset}`
  );
}

export async function getSession(
  sessionId: string
): Promise<import('../types/api').SessionResponse> {
  return fetchJson<import('../types/api').SessionResponse>(`${API_BASE}/sessions/${sessionId}`);
}

export async function deleteSession(
  sessionId: string
): Promise<{ status: string; session_id: string }> {
  return fetchJson(`${API_BASE}/sessions/${sessionId}`, {
    method: 'DELETE',
  });
}

export async function sendMessage(
  sessionId: string,
  content: string,
  executionPolicy?: {
    max_retries_per_step?: number;
    retry_delay_seconds?: number;
    rollback_on_failure?: boolean;
  }
): Promise<import('../types/api').MessageResponse> {
  const request: import('../types/api').MessageRequest = {
    content,
    ...(executionPolicy && {
      execution_policy: {
        max_retries_per_step: executionPolicy.max_retries_per_step ?? 0,
        retry_delay_seconds: executionPolicy.retry_delay_seconds ?? 0.0,
        rollback_on_failure: executionPolicy.rollback_on_failure ?? false,
      },
    }),
  };

  return fetchJson<import('../types/api').MessageResponse>(
    `${API_BASE}/sessions/${sessionId}/messages`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

export async function approveOperations(
  sessionId: string,
  messageId: string,
  stepIds: number[],
  reason?: string
): Promise<{ message_id: string; execution_result: any }> {
  const request = {
    step_ids: stepIds,
    ...(reason && { reason }),
  };

  return fetchJson(`${API_BASE}/sessions/${sessionId}/messages/${messageId}/approve`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function rejectOperations(
  sessionId: string,
  messageId: string,
  stepIds: number[],
  reason?: string
): Promise<{ status: string; message_id: string }> {
  const request = {
    step_ids: stepIds,
    ...(reason && { reason }),
  };

  return fetchJson(`${API_BASE}/sessions/${sessionId}/messages/${messageId}/reject`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

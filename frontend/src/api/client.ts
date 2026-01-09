/**Thin API client - no retries, no transformation, no business logic.

Errors bubble up. No retries. No transformation.
API is the single source of truth.
*/

import type {
  ExecuteRequest,
  ExecuteResponse,
  PlanRequest,
  PlanResponse,
  RunDetailResponse,
  RunRequest,
  RunResponse,
  RunsListResponse,
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

export async function run(
  goal: string,
  context?: Record<string, unknown>,
  executionPolicy?: {
    max_retries_per_step?: number;
    retry_delay_seconds?: number;
    rollback_on_failure?: boolean;
  }
): Promise<RunResponse> {
  const request: RunRequest = {
    goal,
    ...(context && { context }),
    ...(executionPolicy && {
      execution_policy: {
        max_retries_per_step: executionPolicy.max_retries_per_step ?? 0,
        retry_delay_seconds: executionPolicy.retry_delay_seconds ?? 0.0,
        rollback_on_failure: executionPolicy.rollback_on_failure ?? false,
      },
    }),
  };

  return fetchJson<RunResponse>(`${API_BASE}/run`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function listRuns(
  limit: number = 20,
  offset: number = 0
): Promise<RunsListResponse> {
  return fetchJson<RunsListResponse>(
    `${API_BASE}/runs?limit=${limit}&offset=${offset}`
  );
}

export async function getRun(runId: string): Promise<RunDetailResponse> {
  return fetchJson<RunDetailResponse>(`${API_BASE}/runs/${runId}`);
}

export async function approveRun(
  runId: string,
  approvedBy: string,
  reason?: string,
  executionPolicy?: {
    max_retries_per_step?: number;
    retry_delay_seconds?: number;
    rollback_on_failure?: boolean;
  }
): Promise<{ run_id: string; approval_status: string; execution_result: any }> {
  const request: any = {
    approved_by: approvedBy,
    ...(reason && { reason }),
    ...(executionPolicy && {
      execution_policy: {
        max_retries_per_step: executionPolicy.max_retries_per_step ?? 0,
        retry_delay_seconds: executionPolicy.retry_delay_seconds ?? 0.0,
        rollback_on_failure: executionPolicy.rollback_on_failure ?? false,
      },
    }),
  };

  return fetchJson(`${API_BASE}/runs/${runId}/approve`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

export async function rejectRun(
  runId: string,
  rejectedBy: string,
  reason: string
): Promise<{ status: string; run_id: string }> {
  const request = {
    rejected_by: rejectedBy,
    reason,
  };

  return fetchJson(`${API_BASE}/runs/${runId}/reject`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
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

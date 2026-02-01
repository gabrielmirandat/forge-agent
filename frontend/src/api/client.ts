/**Thin API client - no retries, no transformation, no business logic.

Errors bubble up. No retries. No transformation.
API is the single source of truth.
*/

// Removed PlanRequest, PlanResponse, ExecuteRequest, ExecuteResponse - no longer used

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

// Removed plan() and execute() functions - no longer used with direct tool calling

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
  content: string
): Promise<import('../types/api').MessageResponse> {
  const request: import('../types/api').MessageRequest = {
    content,
  };

  return fetchJson<import('../types/api').MessageResponse>(
    `${API_BASE}/sessions/${sessionId}/messages`,
    {
      method: 'POST',
      body: JSON.stringify(request),
    }
  );
}

// Removed approveOperations and rejectOperations - no longer needed with direct tool calling

// Config API
export async function getLLMConfig(): Promise<{
  provider: string;
  model: string;
  temperature: number;
  max_tokens: number;
  timeout: number;
  base_url?: string;
  compression?: string;
  profiling_mode?: boolean;
}> {
  return fetchJson(`${API_BASE}/config/llm`);
}

export async function listLLMProviders(): Promise<{
  providers: Array<{
    id: string;
    name: string;
    description: string;
    config_file: string | null;
    model?: string;
    status?: string;  // ✅, ❌, or ⚠️
    required_fields?: string[];
    optional_fields?: string[];
  }>;
}> {
  return fetchJson(`${API_BASE}/config/llm/providers`);
}

export async function updateLLMConfig(
  config: {
    provider: string;
    model: string;
    temperature?: number;
    max_tokens?: number;
    timeout?: number;
    base_url?: string;
    compression?: string;
    profiling_mode?: boolean;
  }
): Promise<{ status: string; message: string; config: any }> {
  return fetchJson(`${API_BASE}/config/llm`, {
    method: 'PUT',
    body: JSON.stringify(config),
  });
}

export async function switchLLMProvider(
  provider: string,
  configFile?: string
): Promise<{ status: string; message: string; config: any }> {
  const params = new URLSearchParams({ provider });
  if (configFile) {
    params.append('config_file', configFile);
  }
  return fetchJson(`${API_BASE}/config/llm/switch?${params.toString()}`, {
    method: 'POST',
  });
}

export async function restartOllama(): Promise<{
  status: string;
  message: string;
  port?: number;
}> {
  return fetchJson(`${API_BASE}/config/llm/restart`, {
    method: 'POST',
  });
}

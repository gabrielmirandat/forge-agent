/**TypeScript types for API responses.

These types mirror the API schemas exactly.
No transformation, no inference.
*/

// Session types
export interface CreateSessionRequest {
  title?: string;
}

export interface CreateSessionResponse {
  session_id: string;
  title: string;
}

export interface MessageRequest {
  content: string;
}

export interface MessageResponse {
  message_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: number;
}

export interface SessionResponse {
  session_id: string;
  title: string;
  messages: MessageResponse[];
  created_at: number;
  updated_at: number;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
}

export interface SessionsListResponse {
  sessions: SessionSummary[];
  limit: number;
  offset: number;
}

// Router / model tier types
export interface RouterTierStatus {
  selected_model: string | null;
  preferred_models: string[];
  available: string[];
  missing: string[];
  pull_command: string | null;
}

export interface RouterTiersResponse {
  enabled: boolean;
  tiers: Record<string, RouterTierStatus>;
}

export interface RouterTierConfig {
  model: string;
  description: string;
  preferred_models: string[];
}

export interface RouterConfig {
  enabled: boolean;
  default_tier: string;
  tiers: Record<string, RouterTierConfig>;
  rules: Array<{ tier: string; keywords: string[] }>;
}

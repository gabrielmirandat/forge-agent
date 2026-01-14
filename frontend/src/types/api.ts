/**TypeScript types for API responses.

These types mirror the API schemas exactly.
No transformation, no inference.
*/

export interface PlanRequest {
  goal: string;
  context?: Record<string, unknown>;
}

export interface PlanResponse {
  plan: Plan;
  diagnostics: PlannerDiagnostics;
}

export interface ExecuteRequest {
  plan: Plan;
  execution_policy?: ExecutionPolicy;
}

export interface ExecuteResponse {
  execution_result: ExecutionResult;
}

// Core types

export interface Plan {
  plan_id: string;
  objective: string;
  steps: PlanStep[];
  estimated_time_seconds?: number;
  notes?: string;
}

export interface PlanStep {
  step_id: number;
  tool: string;
  operation: string;
  arguments: Record<string, unknown>;
  rationale: string;
}

export interface PlannerDiagnostics {
  model_name: string;
  temperature: number;
  retries_used: number;
  raw_llm_response: string;
  extracted_json?: string | null;
  validation_errors?: string[] | null;
}

export interface ExecutionPolicy {
  max_retries_per_step: number;
  retry_delay_seconds: number;
  rollback_on_failure: boolean;
}

export interface ExecutionResult {
  plan_id: string;
  objective: string;
  steps: StepExecutionResult[];
  success: boolean;
  stopped_at_step?: number | null;
  rollback_attempted: boolean;
  rollback_success?: boolean | null;
  rollback_steps: RollbackStepResult[];
  started_at: number;
  finished_at: number;
}

export interface StepExecutionResult {
  step_id: number;
  tool: string;
  operation: string;
  arguments: Record<string, unknown>;
  success: boolean;
  output?: unknown;
  error?: string | null;
  retries_attempted: number;
  started_at: number;
  finished_at: number;
}

export interface RollbackStepResult {
  step_id: number;
  tool: string;
  operation: string;
  success: boolean;
  error?: string | null;
  started_at: number;
  finished_at: number;
}

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
  execution_policy?: ExecutionPolicy;
}

export interface MessageResponse {
  message_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  created_at: number;
  plan_result?: {
    plan: Plan;
    diagnostics: PlannerDiagnostics;
  } | null;
  execution_result?: ExecutionResult | null;
  pending_approval_steps?: number[] | null;
  restricted_steps?: number[] | null;
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

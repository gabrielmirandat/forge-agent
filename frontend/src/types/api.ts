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

export interface RunRequest {
  goal: string;
  context?: Record<string, unknown>;
  execution_policy?: ExecutionPolicy;
}

export interface RunResponse {
  plan_result: {
    plan: Plan;
    diagnostics: PlannerDiagnostics;
  };
  execution_result: ExecutionResult | null; // null if pending approval
}

export interface RunsListResponse {
  runs: RunSummary[];
  limit: number;
  offset: number;
}

export interface RunDetailResponse {
  run: RunRecord;
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

export interface RunSummary {
  run_id: string;
  plan_id: string;
  objective: string;
  success: boolean;
  created_at: number;
}

export type ApprovalStatus = 'pending' | 'approved' | 'rejected';

export interface RunRecord {
  run_id: string;
  plan_id: string;
  objective: string;
  plan_result: {
    plan: Plan;
    diagnostics: PlannerDiagnostics;
  };
  execution_result: ExecutionResult | null; // null if not executed
  created_at: number;
  approval_status: ApprovalStatus;
  approval_reason?: string | null;
  approved_at?: number | null;
  approved_by?: string | null;
}


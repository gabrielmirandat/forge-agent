import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export interface GoalRequest {
  goal: string
  context?: Record<string, unknown>
  repo_path?: string
}

export interface PlanResponse {
  plan_id: string
  steps: Array<{
    tool: string
    operation: string
    parameters: Record<string, unknown>
  }>
  estimated_time?: number
}

export interface ExecutionResult {
  execution_id: string
  status: string
  results: Array<Record<string, unknown>>
  error?: string
}

export const agentApi = {
  createGoal: async (request: GoalRequest): Promise<PlanResponse> => {
    const response = await apiClient.post<PlanResponse>('/api/v1/goals', request)
    return response.data
  },

  executePlan: async (planId: string): Promise<ExecutionResult> => {
    const response = await apiClient.post<ExecutionResult>(
      `/api/v1/plans/${planId}/execute`
    )
    return response.data
  },

  getStatus: async () => {
    const response = await apiClient.get('/api/v1/status')
    return response.data
  },
}


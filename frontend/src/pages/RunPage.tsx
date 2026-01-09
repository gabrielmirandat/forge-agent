/**Run page - create a new run.

No auto-refresh, no polling, no optimistic UI.
Displays exactly what the API returns.
*/

import { useState } from 'react';
import { run } from '../api/client';
import { DiagnosticsViewer } from '../components/DiagnosticsViewer';
import { ExecutionViewer } from '../components/ExecutionViewer';
import { JsonBlock } from '../components/JsonBlock';
import { PlanViewer } from '../components/PlanViewer';
import type { RunResponse } from '../types/api';

export function RunPage() {
  const [goal, setGoal] = useState('');
  const [context, setContext] = useState('{}');
  const [executionPolicy, setExecutionPolicy] = useState('{}');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunResponse | null>(null);

  const handleRun = async () => {
    if (!goal.trim()) {
      setError('Goal is required');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let contextObj: Record<string, unknown> | undefined;
      try {
        contextObj = JSON.parse(context);
      } catch (e) {
        setError(`Invalid context JSON: ${e}`);
        setLoading(false);
        return;
      }

      let executionPolicyObj: {
        max_retries_per_step?: number;
        retry_delay_seconds?: number;
        rollback_on_failure?: boolean;
      } | undefined;
      try {
        const parsed = JSON.parse(executionPolicy);
        if (Object.keys(parsed).length > 0) {
          executionPolicyObj = parsed;
        }
      } catch (e) {
        setError(`Invalid execution policy JSON: ${e}`);
        setLoading(false);
        return;
      }

      const response = await run(goal, contextObj, executionPolicyObj);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
      <h1>Create Run</h1>

      <div style={{ marginBottom: '2rem' }}>
        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
            Goal *
          </label>
          <textarea
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="Enter goal description..."
            style={{
              width: '100%',
              minHeight: '100px',
              padding: '0.5rem',
              fontSize: '1rem',
              fontFamily: 'monospace',
            }}
          />
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
            Context (JSON, optional)
          </label>
          <textarea
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder='{"key": "value"}'
            style={{
              width: '100%',
              minHeight: '80px',
              padding: '0.5rem',
              fontSize: '0.875rem',
              fontFamily: 'monospace',
            }}
          />
        </div>

        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
            Execution Policy (JSON, optional)
          </label>
          <textarea
            value={executionPolicy}
            onChange={(e) => setExecutionPolicy(e.target.value)}
            placeholder='{"max_retries_per_step": 2, "retry_delay_seconds": 1.0, "rollback_on_failure": false}'
            style={{
              width: '100%',
              minHeight: '80px',
              padding: '0.5rem',
              fontSize: '0.875rem',
              fontFamily: 'monospace',
            }}
          />
        </div>

        <button
          onClick={handleRun}
          disabled={loading}
          style={{
            padding: '0.75rem 1.5rem',
            fontSize: '1rem',
            fontWeight: 'bold',
            background: loading ? '#ccc' : '#007bff',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Running...' : 'Run'}
        </button>
      </div>

      {error && (
        <div
          style={{
            padding: '1rem',
            marginBottom: '1rem',
            background: '#f8d7da',
            border: '1px solid #f5c6cb',
            borderRadius: '4px',
            color: '#721c24',
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div>
          <h2>Results</h2>

          {result.execution_result === null && (
            <div
              style={{
                padding: '1rem',
                marginBottom: '2rem',
                background: '#fff3cd',
                border: '1px solid #ffc107',
                borderRadius: '4px',
              }}
            >
              <strong>🕒 Awaiting Approval</strong>
              <div style={{ marginTop: '0.5rem' }}>
                Plan has been generated. Execution will occur after approval.
              </div>
            </div>
          )}

          <div style={{ marginBottom: '2rem' }}>
            <h3>Plan</h3>
            <PlanViewer plan={result.plan_result.plan} />
          </div>

          <div style={{ marginBottom: '2rem' }}>
            <DiagnosticsViewer diagnostics={result.plan_result.diagnostics} />
          </div>

          {result.execution_result && (
            <div style={{ marginBottom: '2rem' }}>
              <h3>Execution</h3>
              <ExecutionViewer executionResult={result.execution_result} />
            </div>
          )}

          <div>
            <h3>Raw Response</h3>
            <JsonBlock data={result} title="Complete Response" defaultCollapsed={true} />
          </div>
        </div>
      )}
    </div>
  );
}


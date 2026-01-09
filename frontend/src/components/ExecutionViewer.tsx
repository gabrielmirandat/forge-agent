/**Execution viewer component - renders execution results.

Displays exactly what the API returns. No interpretation.
Failures are shown explicitly, not as UI errors.
*/

import type { ExecutionResult } from '../types/api';

interface ExecutionViewerProps {
  executionResult: ExecutionResult;
}

export function ExecutionViewer({ executionResult }: ExecutionViewerProps) {
  const duration = executionResult.finished_at - executionResult.started_at;

  return (
    <div>
      <div
        style={{
          padding: '1rem',
          marginBottom: '1rem',
          borderRadius: '4px',
          background: executionResult.success ? '#d4edda' : '#f8d7da',
          border: `1px solid ${executionResult.success ? '#c3e6cb' : '#f5c6cb'}`,
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <strong>
            Execution {executionResult.success ? 'Succeeded' : 'Failed'}
          </strong>
          <span style={{ fontSize: '0.875rem', color: '#666' }}>
            Duration: {duration.toFixed(3)}s
          </span>
        </div>
        {executionResult.stopped_at_step && (
          <div style={{ marginTop: '0.5rem', color: '#721c24' }}>
            Stopped at step: {executionResult.stopped_at_step}
          </div>
        )}
      </div>

      {executionResult.steps.length === 0 ? (
        <div style={{ padding: '1rem', background: '#fff3cd', border: '1px solid #ffc107', borderRadius: '4px' }}>
          No steps executed (empty plan)
        </div>
      ) : (
        <div>
          <h3>Executed Steps ({executionResult.steps.length})</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {executionResult.steps.map((step) => {
              const stepDuration = step.finished_at - step.started_at;
              return (
                <div
                  key={step.step_id}
                  style={{
                    border: `1px solid ${step.success ? '#c3e6cb' : '#f5c6cb'}`,
                    borderRadius: '4px',
                    padding: '1rem',
                    background: step.success ? '#d4edda' : '#f8d7da',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <strong>Step {step.step_id}</strong>
                    <div style={{ display: 'flex', gap: '1rem', fontSize: '0.875rem', color: '#666' }}>
                      <span>{step.tool}.{step.operation}</span>
                      <span>Duration: {stepDuration.toFixed(3)}s</span>
                      {step.retries_attempted > 0 && (
                        <span>Retries: {step.retries_attempted}</span>
                      )}
                    </div>
                  </div>

                  {step.success ? (
                    <div>
                      <strong>Output:</strong>
                      <pre
                        style={{
                          marginTop: '0.5rem',
                          padding: '0.5rem',
                          background: '#fff',
                          borderRadius: '4px',
                          fontSize: '0.875rem',
                          overflow: 'auto',
                        }}
                      >
                        {JSON.stringify(step.output, null, 2)}
                      </pre>
                    </div>
                  ) : (
                    <div style={{ color: '#721c24' }}>
                      <strong>Error:</strong> {step.error || 'Unknown error'}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {executionResult.rollback_attempted && (
        <div style={{ marginTop: '2rem' }}>
          <h3>Rollback</h3>
          <div
            style={{
              padding: '1rem',
              marginBottom: '1rem',
              borderRadius: '4px',
              background: executionResult.rollback_success ? '#d4edda' : '#f8d7da',
              border: `1px solid ${executionResult.rollback_success ? '#c3e6cb' : '#f5c6cb'}`,
            }}
          >
            <strong>
              Rollback {executionResult.rollback_success ? 'Succeeded' : 'Failed'}
            </strong>
          </div>
          {executionResult.rollback_steps.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {executionResult.rollback_steps.map((rollback) => {
                const rollbackDuration = rollback.finished_at - rollback.started_at;
                return (
                  <div
                    key={rollback.step_id}
                    style={{
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      padding: '0.75rem',
                      background: '#fff',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>
                        Step {rollback.step_id} - {rollback.tool}.{rollback.operation}
                      </span>
                      <span style={{ fontSize: '0.875rem', color: '#666' }}>
                        {rollback.success ? '✓' : '✗'} ({rollbackDuration.toFixed(3)}s)
                      </span>
                    </div>
                    {rollback.error && (
                      <div style={{ marginTop: '0.5rem', color: '#721c24', fontSize: '0.875rem' }}>
                        {rollback.error}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


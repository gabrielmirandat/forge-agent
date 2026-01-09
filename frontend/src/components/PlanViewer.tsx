/**Plan viewer component - renders plan steps in order.

Displays exactly what the API returns. No interpretation.
*/

import type { Plan } from '../types/api';

interface PlanViewerProps {
  plan: Plan;
}

export function PlanViewer({ plan }: PlanViewerProps) {
  if (plan.steps.length === 0) {
    return (
      <div style={{ padding: '1rem', background: '#fff3cd', border: '1px solid #ffc107', borderRadius: '4px' }}>
        <strong>Empty Plan</strong>
        {plan.notes && (
          <div style={{ marginTop: '0.5rem' }}>
            <strong>Notes:</strong> {plan.notes}
          </div>
        )}
      </div>
    );
  }

  return (
    <div>
      <h3>Plan Steps ({plan.steps.length})</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {plan.steps.map((step) => (
          <div
            key={step.step_id}
            style={{
              border: '1px solid #ddd',
              borderRadius: '4px',
              padding: '1rem',
              background: '#fff',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
              <strong>Step {step.step_id}</strong>
              <span style={{ color: '#666' }}>
                {step.tool}.{step.operation}
              </span>
            </div>
            <div style={{ marginBottom: '0.5rem' }}>
              <strong>Rationale:</strong> {step.rationale}
            </div>
            {Object.keys(step.arguments).length > 0 && (
              <div>
                <strong>Arguments:</strong>
                <pre
                  style={{
                    marginTop: '0.5rem',
                    padding: '0.5rem',
                    background: '#f5f5f5',
                    borderRadius: '4px',
                    fontSize: '0.875rem',
                    overflow: 'auto',
                  }}
                >
                  {JSON.stringify(step.arguments, null, 2)}
                </pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}


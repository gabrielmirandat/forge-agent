/**Diagnostics viewer component - displays planner diagnostics.

Shows exactly what the API returns. No interpretation.
*/

import type { PlannerDiagnostics } from '../types/api';
import { JsonBlock } from './JsonBlock';

interface DiagnosticsViewerProps {
  diagnostics: PlannerDiagnostics;
}

export function DiagnosticsViewer({ diagnostics }: DiagnosticsViewerProps) {
  return (
    <div>
      <h3>Planner Diagnostics</h3>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
        <div>
          <strong>Model:</strong> {diagnostics.model_name}
        </div>
        <div>
          <strong>Temperature:</strong> {diagnostics.temperature}
        </div>
        <div>
          <strong>Retries Used:</strong> {diagnostics.retries_used}
        </div>
      </div>

      {diagnostics.validation_errors && diagnostics.validation_errors.length > 0 && (
        <div
          style={{
            padding: '1rem',
            marginBottom: '1rem',
            background: '#f8d7da',
            border: '1px solid #f5c6cb',
            borderRadius: '4px',
          }}
        >
          <strong>Validation Errors:</strong>
          <ul style={{ marginTop: '0.5rem', marginBottom: 0 }}>
            {diagnostics.validation_errors.map((error, idx) => (
              <li key={idx}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      <div style={{ marginBottom: '1rem' }}>
        <strong>Raw LLM Response:</strong>
        <pre
          style={{
            marginTop: '0.5rem',
            padding: '0.5rem',
            background: '#f5f5f5',
            borderRadius: '4px',
            fontSize: '0.875rem',
            overflow: 'auto',
            maxHeight: '200px',
          }}
        >
          {diagnostics.raw_llm_response}
        </pre>
      </div>

      {diagnostics.extracted_json && (
        <div>
          <strong>Extracted JSON:</strong>
          <JsonBlock
            data={JSON.parse(diagnostics.extracted_json)}
            title="Extracted JSON"
            defaultCollapsed={true}
          />
        </div>
      )}
    </div>
  );
}


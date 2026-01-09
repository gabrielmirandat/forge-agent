/**Run detail page - inspect a single run.

Displays exactly what the API returns. No interpretation.
Every step must be inspectable.
*/

import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getRun } from '../api/client';
import { ApprovalPanel } from '../components/ApprovalPanel';
import { DiagnosticsViewer } from '../components/DiagnosticsViewer';
import { ExecutionViewer } from '../components/ExecutionViewer';
import { JsonBlock } from '../components/JsonBlock';
import { PlanViewer } from '../components/PlanViewer';
import type { RunRecord } from '../types/api';

export function RunDetailPage() {
  const { runId } = useParams<{ runId: string }>();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [run, setRun] = useState<RunRecord | null>(null);

  useEffect(() => {
    if (!runId) {
      setError('Run ID is required');
      setLoading(false);
      return;
    }

    const loadRun = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await getRun(runId);
        setRun(response.run);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadRun();
  }, [runId]);

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  if (loading) {
    return (
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        <div>Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        <div
          style={{
            padding: '1rem',
            background: '#f8d7da',
            border: '1px solid #f5c6cb',
            borderRadius: '4px',
            color: '#721c24',
          }}
        >
          <strong>Error:</strong> {error}
        </div>
      </div>
    );
  }

  if (!run) {
    return (
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
        <div>Run not found</div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
      <h1>Run Details</h1>

      <div style={{ marginBottom: '2rem', padding: '1rem', background: '#f5f5f5', borderRadius: '4px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <strong>Run ID:</strong> {run.run_id}
          </div>
          <div>
            <strong>Plan ID:</strong> {run.plan_id}
          </div>
          <div>
            <strong>Objective:</strong> {run.objective}
          </div>
          <div>
            <strong>Created At:</strong> {formatDate(run.created_at)}
          </div>
          <div>
            <strong>Approval Status:</strong>{' '}
            <span
              style={{
                padding: '0.25rem 0.5rem',
                borderRadius: '4px',
                background:
                  run.approval_status === 'approved'
                    ? '#d4edda'
                    : run.approval_status === 'rejected'
                    ? '#f8d7da'
                    : '#fff3cd',
                color:
                  run.approval_status === 'approved'
                    ? '#155724'
                    : run.approval_status === 'rejected'
                    ? '#721c24'
                    : '#856404',
              }}
            >
              {run.approval_status === 'pending' && '🕒 Pending'}
              {run.approval_status === 'approved' && '✅ Approved'}
              {run.approval_status === 'rejected' && '❌ Rejected'}
            </span>
          </div>
        </div>
      </div>

      <ApprovalPanel
        run={run}
        onUpdate={async () => {
          const response = await getRun(run.run_id);
          setRun(response.run);
        }}
      />

      <div style={{ marginBottom: '2rem' }}>
        <h2>Plan</h2>
        <PlanViewer plan={run.plan_result.plan} />
      </div>

      <div style={{ marginBottom: '2rem' }}>
        <h2>Planner Diagnostics</h2>
        <DiagnosticsViewer diagnostics={run.plan_result.diagnostics} />
      </div>

      {run.execution_result ? (
        <div style={{ marginBottom: '2rem' }}>
          <h2>Execution Result</h2>
          <ExecutionViewer executionResult={run.execution_result} />
        </div>
      ) : (
        <div style={{ marginBottom: '2rem' }}>
          <h2>Execution Result</h2>
          <div
            style={{
              padding: '1rem',
              background: '#f5f5f5',
              border: '1px solid #ddd',
              borderRadius: '4px',
            }}
          >
            Execution has not occurred yet.
            {run.approval_status === 'pending' && ' Awaiting approval.'}
            {run.approval_status === 'rejected' && ' Run was rejected.'}
          </div>
        </div>
      )}

      <div>
        <h2>Raw Data</h2>
        <JsonBlock data={run} title="Complete Run Record" defaultCollapsed={true} />
      </div>
    </div>
  );
}


/**Approval panel component - approve/reject runs.

No auto-approve, no auto-execute.
Approval requires explicit click.
Rejection requires reason.
*/

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { approveRun, rejectRun } from '../api/client';
import type { ApprovalStatus, RunRecord } from '../types/api';

interface ApprovalPanelProps {
  run: RunRecord;
  onUpdate: () => void;
}

export function ApprovalPanel({ run, onUpdate }: ApprovalPanelProps) {
  const [approving, setApproving] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [approvedBy, setApprovedBy] = useState('');
  const [rejectionReason, setRejectionReason] = useState('');
  const navigate = useNavigate();

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const handleApprove = async () => {
    if (!approvedBy.trim()) {
      setError('Approver name is required');
      return;
    }

    setApproving(true);
    setError(null);

    try {
      const result = await approveRun(run.run_id, approvedBy);
      // Reload the run to get updated data
      onUpdate();
      // Optionally navigate to show execution result
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to approve');
    } finally {
      setApproving(false);
    }
  };

  const handleReject = async () => {
    if (!approvedBy.trim()) {
      setError('Rejector name is required');
      return;
    }
    if (!rejectionReason.trim()) {
      setError('Rejection reason is required');
      return;
    }

    setRejecting(true);
    setError(null);

    try {
      await rejectRun(run.run_id, approvedBy, rejectionReason);
      // Reload the run to show rejection status
      onUpdate();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to reject');
    } finally {
      setRejecting(false);
    }
  };

  if (run.approval_status === 'pending') {
    return (
      <div
        style={{
          padding: '1.5rem',
          marginBottom: '2rem',
          background: '#fff3cd',
          border: '1px solid #ffc107',
          borderRadius: '4px',
        }}
      >
        <h3>🕒 Pending Approval</h3>
        <p>This run requires approval before execution.</p>

        {error && (
          <div
            style={{
              padding: '0.75rem',
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

        <div style={{ marginBottom: '1rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
            Your Name/Email *
          </label>
          <input
            type="text"
            value={approvedBy}
            onChange={(e) => setApprovedBy(e.target.value)}
            placeholder="Enter your name or email"
            style={{
              width: '100%',
              padding: '0.5rem',
              fontSize: '1rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: '1rem' }}>
          <button
            onClick={handleApprove}
            disabled={approving || rejecting}
            style={{
              padding: '0.75rem 1.5rem',
              fontSize: '1rem',
              fontWeight: 'bold',
              background: approving ? '#ccc' : '#28a745',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: approving || rejecting ? 'not-allowed' : 'pointer',
            }}
          >
            {approving ? 'Approving...' : '✓ Approve'}
          </button>

          <div style={{ flex: 1 }}>
            <input
              type="text"
              value={rejectionReason}
              onChange={(e) => setRejectionReason(e.target.value)}
              placeholder="Rejection reason (required for reject)"
              style={{
                width: '100%',
                padding: '0.5rem',
                fontSize: '1rem',
                border: '1px solid #ccc',
                borderRadius: '4px',
              }}
            />
          </div>

          <button
            onClick={handleReject}
            disabled={rejecting || approving}
            style={{
              padding: '0.75rem 1.5rem',
              fontSize: '1rem',
              fontWeight: 'bold',
              background: rejecting ? '#ccc' : '#dc3545',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: rejecting || approving ? 'not-allowed' : 'pointer',
            }}
          >
            {rejecting ? 'Rejecting...' : '✗ Reject'}
          </button>
        </div>
      </div>
    );
  }

  if (run.approval_status === 'approved') {
    return (
      <div
        style={{
          padding: '1rem',
          marginBottom: '2rem',
          background: '#d4edda',
          border: '1px solid #c3e6cb',
          borderRadius: '4px',
        }}
      >
        <h3>✅ Approved</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <strong>Approved by:</strong> {run.approved_by || 'Unknown'}
          </div>
          {run.approved_at && (
            <div>
              <strong>Approved at:</strong> {formatDate(run.approved_at)}
            </div>
          )}
          {run.approval_reason && (
            <div style={{ gridColumn: '1 / -1' }}>
              <strong>Reason:</strong> {run.approval_reason}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (run.approval_status === 'rejected') {
    return (
      <div
        style={{
          padding: '1rem',
          marginBottom: '2rem',
          background: '#f8d7da',
          border: '1px solid #f5c6cb',
          borderRadius: '4px',
        }}
      >
        <h3>❌ Rejected</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <strong>Rejected by:</strong> {run.approved_by || 'Unknown'}
          </div>
          {run.approved_at && (
            <div>
              <strong>Rejected at:</strong> {formatDate(run.approved_at)}
            </div>
          )}
          {run.approval_reason && (
            <div style={{ gridColumn: '1 / -1' }}>
              <strong>Reason:</strong> {run.approval_reason}
            </div>
          )}
        </div>
      </div>
    );
  }

  return null;
}


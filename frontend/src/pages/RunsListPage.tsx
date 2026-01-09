/**Runs list page - browse historical runs.

No auto-refresh, no polling.
Displays exactly what the API returns.
*/

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { listRuns } from '../api/client';
import type { RunSummary } from '../types/api';

// Note: RunSummary doesn't include approval_status yet
// This would need to be added to the API response

export function RunsListPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [limit] = useState(20);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  const loadRuns = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await listRuns(limit, offset);
      setRuns(response.runs);
      setHasMore(response.runs.length === limit);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadRuns();
  }, [offset]);

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
      <h1>Run History</h1>

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

      {loading ? (
        <div>Loading...</div>
      ) : runs.length === 0 ? (
        <div>No runs found.</div>
      ) : (
        <>
          <table
            style={{
              width: '100%',
              borderCollapse: 'collapse',
              marginBottom: '1rem',
            }}
          >
            <thead>
              <tr style={{ background: '#f5f5f5' }}>
                <th style={{ padding: '0.75rem', textAlign: 'left', border: '1px solid #ddd' }}>
                  Run ID
                </th>
                <th style={{ padding: '0.75rem', textAlign: 'left', border: '1px solid #ddd' }}>
                  Objective
                </th>
                <th style={{ padding: '0.75rem', textAlign: 'center', border: '1px solid #ddd' }}>
                  Success
                </th>
                <th style={{ padding: '0.75rem', textAlign: 'left', border: '1px solid #ddd' }}>
                  Created At
                </th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id}>
                  <td style={{ padding: '0.75rem', border: '1px solid #ddd' }}>
                    <Link
                      to={`/runs/${run.run_id}`}
                      style={{ color: '#007bff', textDecoration: 'none' }}
                    >
                      {run.run_id.substring(0, 8)}...
                    </Link>
                  </td>
                  <td style={{ padding: '0.75rem', border: '1px solid #ddd' }}>
                    {run.objective.length > 60
                      ? `${run.objective.substring(0, 60)}...`
                      : run.objective}
                  </td>
                  <td
                    style={{
                      padding: '0.75rem',
                      border: '1px solid #ddd',
                      textAlign: 'center',
                    }}
                  >
                    <span
                      style={{
                        padding: '0.25rem 0.5rem',
                        borderRadius: '4px',
                        background: run.success ? '#d4edda' : '#f8d7da',
                        color: run.success ? '#155724' : '#721c24',
                        fontSize: '0.875rem',
                      }}
                    >
                      {run.success ? '✓' : '✗'}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem', border: '1px solid #ddd' }}>
                    {formatDate(run.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <button
              onClick={() => setOffset(Math.max(0, offset - limit))}
              disabled={offset === 0}
              style={{
                padding: '0.5rem 1rem',
                background: offset === 0 ? '#ccc' : '#007bff',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: offset === 0 ? 'not-allowed' : 'pointer',
              }}
            >
              Previous
            </button>
            <span>
              Showing {offset + 1}-{offset + runs.length}
            </span>
            <button
              onClick={() => setOffset(offset + limit)}
              disabled={!hasMore}
              style={{
                padding: '0.5rem 1rem',
                background: !hasMore ? '#ccc' : '#007bff',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: !hasMore ? 'not-allowed' : 'pointer',
              }}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
}


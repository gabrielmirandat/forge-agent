/**JSON block component - pretty-prints JSON with copy-to-clipboard.

No transformation - displays exactly what is passed.
*/

import { useState } from 'react';

interface JsonBlockProps {
  data: unknown;
  title?: string;
  defaultCollapsed?: boolean;
}

export function JsonBlock({
  data,
  title,
  defaultCollapsed = false,
}: JsonBlockProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [copied, setCopied] = useState(false);

  const jsonString = JSON.stringify(data, null, 2);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(jsonString);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div style={{ marginBottom: '1rem', border: '1px solid #ccc', borderRadius: '4px' }}>
      {(title || true) && (
        <div
          style={{
            padding: '0.5rem',
            background: '#f5f5f5',
            borderBottom: '1px solid #ccc',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            cursor: 'pointer',
          }}
          onClick={() => setCollapsed(!collapsed)}
        >
          <strong>{title || 'JSON'}</strong>
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleCopy();
              }}
              style={{
                padding: '0.25rem 0.5rem',
                fontSize: '0.875rem',
                cursor: 'pointer',
              }}
            >
              {copied ? 'Copied!' : 'Copy'}
            </button>
            <span>{collapsed ? '▼' : '▲'}</span>
          </div>
        </div>
      )}
      {!collapsed && (
        <pre
          style={{
            padding: '1rem',
            margin: 0,
            overflow: 'auto',
            background: '#fafafa',
            fontSize: '0.875rem',
            fontFamily: 'monospace',
          }}
        >
          {jsonString}
        </pre>
      )}
    </div>
  );
}


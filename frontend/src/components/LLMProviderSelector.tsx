/**LLM Provider Selector - Switch between LLM providers at runtime.*/

import { useEffect, useState } from 'react';
import * as api from '../api/client';

interface LLMConfig {
  provider: string;
  model: string;
  temperature: number;
  max_tokens: number;
  timeout: number;
  base_url?: string;
  compression?: string;
  profiling_mode?: boolean;
}

interface Provider {
  id: string;
  name: string;
  description: string;
  config_file: string | null;
  model?: string;
  status?: string;  // ✅, ❌, or ⚠️
}

export function LLMProviderSelector() {
  const [currentConfig, setCurrentConfig] = useState<LLMConfig | null>(null);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    loadConfig();
    loadProviders();
  }, []);

  async function loadConfig() {
    try {
      const config = await api.getLLMConfig();
      setCurrentConfig(config);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load config');
    }
  }

  async function loadProviders() {
    try {
      const response = await api.listLLMProviders();
      setProviders(response.providers);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load providers');
    }
  }

  async function handleSwitchProvider(providerId: string) {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      console.log('Switching to provider:', providerId);
      const result = await api.switchLLMProvider(providerId);
      console.log('Switch result:', result);
      const provider = providers.find((p) => p.id === providerId);
      setSuccess(`Switched to ${provider?.name || providerId}`);
      // Reload config to get updated model
      await loadConfig();
      // Reload providers to ensure we have latest status
      await loadProviders();
      // Auto-hide success message after 3 seconds
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      console.error('Failed to switch provider:', err);
      setError(err instanceof Error ? err.message : 'Failed to switch provider');
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
    }
  }

  if (!currentConfig) {
    return (
      <div style={{ color: '#888', fontSize: '0.75rem', textAlign: 'center', padding: '0.5rem' }}>
        Loading...
      </div>
    );
  }

  // Find provider by model name, not provider ID, since provider is always "ollama"
  // but provider.id can be "ollama", "ollama.qwen", "ollama.mistral", etc.
  const currentProvider = providers.find((p) => p.model === currentConfig.model);

  return (
    <div>
      {/* Current provider display - always visible */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          padding: '0.75rem',
          background: expanded ? '#2d2d30' : 'transparent',
          borderRadius: '6px',
          cursor: 'pointer',
          border: '1px solid #444',
          transition: 'all 0.2s',
        }}
        onMouseEnter={(e) => {
          if (!expanded) {
            e.currentTarget.style.background = '#2d2d30';
          }
        }}
        onMouseLeave={(e) => {
          if (!expanded) {
            e.currentTarget.style.background = 'transparent';
          }
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '0.75rem', color: '#888', marginBottom: '0.25rem' }}>
              LLM Provider
            </div>
            <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#fff' }}>
              {currentProvider?.name || currentConfig.provider}
            </div>
            <div style={{ fontSize: '0.7rem', color: '#aaa', marginTop: '0.125rem' }}>
              {currentConfig.model}
            </div>
          </div>
          <div style={{ color: '#888', fontSize: '0.75rem' }}>
            {expanded ? '▼' : '▶'}
          </div>
        </div>
      </div>

      {/* Expanded provider list */}
      {expanded && (
        <div
          style={{
            marginTop: '0.5rem',
            padding: '0.5rem',
            background: '#1a1a1c',
            borderRadius: '6px',
            border: '1px solid #333',
          }}
        >
          {error && (
            <div
              style={{
                padding: '0.5rem',
                background: '#4a1f1f',
                border: '1px solid #6a2f2f',
                borderRadius: '4px',
                color: '#ff6b6b',
                fontSize: '0.75rem',
                marginBottom: '0.5rem',
              }}
            >
              {error}
            </div>
          )}

          {success && (
            <div
              style={{
                padding: '0.5rem',
                background: '#1f4a1f',
                border: '1px solid #2f6a2f',
                borderRadius: '4px',
                color: '#6bff6b',
                fontSize: '0.75rem',
                marginBottom: '0.5rem',
              }}
            >
              {success}
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
            {providers.map((provider) => {
              // Compare by model name, not provider ID, since provider is always "ollama"
              // but provider.id can be "ollama", "ollama.qwen", "ollama.mistral", etc.
              const isActive = currentConfig.model === provider.model;
              return (
                <button
                  key={provider.id}
                  onClick={() => !isActive && handleSwitchProvider(provider.id)}
                  disabled={loading || isActive}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    padding: '0.625rem',
                    borderRadius: '4px',
                    border: isActive ? '1px solid #19c37d' : '1px solid #444',
                    background: isActive ? '#1a3a2a' : 'transparent',
                    color: isActive ? '#19c37d' : '#fff',
                    fontSize: '0.8125rem',
                    cursor: loading || isActive ? 'not-allowed' : 'pointer',
                    opacity: loading || isActive ? 0.7 : 1,
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive && !loading) {
                      e.currentTarget.style.background = '#2d2d30';
                      e.currentTarget.style.borderColor = '#555';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive && !loading) {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.borderColor = '#444';
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.125rem' }}>
                    <div style={{ fontWeight: '500' }}>
                      {provider.name}
                    </div>
                    {provider.status && (
                      <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                        {provider.status}
                      </span>
                    )}
                  </div>
                  {provider.description && (
                    <div style={{ fontSize: '0.7rem', color: '#aaa', marginTop: '0.125rem' }}>
                      {provider.description}
                    </div>
                  )}
                </button>
              );
            })}
          </div>

          {loading && (
            <div
              style={{
                marginTop: '0.5rem',
                textAlign: 'center',
                color: '#888',
                fontSize: '0.75rem',
              }}
            >
              Switching...
            </div>
          )}
        </div>
      )}
    </div>
  );
}

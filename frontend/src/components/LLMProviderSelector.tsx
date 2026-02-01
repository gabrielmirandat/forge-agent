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
  health_status?: string;  // "healthy", "unhealthy", "unknown"
  capabilities?: {
    tool_calling?: boolean;
    image_inputs?: boolean;
    reasoning_output?: boolean;
    max_input_tokens?: number;
    [key: string]: any;
  };
  error?: string;
  selectable?: boolean;  // Whether model can be selected (healthy models only)
}

export function LLMProviderSelector() {
  const [currentConfig, setCurrentConfig] = useState<LLMConfig | null>(null);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [restarting, setRestarting] = useState(false);

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

  async function handleRestartOllama() {
    setRestarting(true);
    setError(null);
    setSuccess(null);

    try {
      await api.restartOllama();
      setSuccess('Ollama restarted successfully');
      // Reload providers after restart to refresh health status
      await loadProviders();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      console.error('Failed to restart Ollama:', err);
      setError(err instanceof Error ? err.message : 'Failed to restart Ollama');
      setTimeout(() => setError(null), 5000);
    } finally {
      setRestarting(false);
    }
  }

  async function handleSwitchProvider(providerId: string) {
    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const provider = providers.find((p) => p.id === providerId);
      console.log('Switching to provider:', {
        providerId,
        providerName: provider?.name,
        expectedModel: provider?.model,
        currentModel: currentConfig?.model
      });
      
      const result = await api.switchLLMProvider(providerId);
      console.log('Switch result:', result);
      
      // Reload config to get updated model - wait a bit for backend to update
      await new Promise(resolve => setTimeout(resolve, 100));
      await loadConfig();
      
      // Verify the model actually changed
      const newConfig = await api.getLLMConfig();
      console.log('After switch - new config:', {
        model: newConfig.model,
        expectedModel: provider?.model,
        match: newConfig.model === provider?.model
      });
      
      if (newConfig.model !== provider?.model) {
        console.warn('Model mismatch after switch:', {
          expected: provider?.model,
          actual: newConfig.model
        });
      }
      
      setSuccess(`Switched to ${provider?.name || providerId}`);
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
  // Use normalized comparison to handle case/whitespace differences
  const currentModel = (currentConfig.model || '').trim().toLowerCase();
  const currentProvider = providers.find((p) => {
    const providerModel = (p.model || '').trim().toLowerCase();
    return providerModel === currentModel;
  });

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
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
              <div style={{ fontSize: '0.75rem', color: '#888' }}>
                LLM Provider
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRestartOllama();
                }}
                disabled={restarting}
                title="Restart Ollama container (refresh model state)"
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: restarting ? '#888' : '#19c37d',
                  cursor: restarting ? 'not-allowed' : 'pointer',
                  padding: '0.125rem 0.375rem',
                  display: 'flex',
                  alignItems: 'center',
                  fontSize: '0.875rem',
                  opacity: restarting ? 0.5 : 1,
                  transition: 'all 0.2s',
                  borderRadius: '4px',
                }}
                onMouseEnter={(e) => {
                  if (!restarting) {
                    e.currentTarget.style.background = '#1a3a2a';
                    e.currentTarget.style.color = '#4ade80';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!restarting) {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.color = '#19c37d';
                  }
                }}
              >
                {restarting ? '⟳' : '↻'}
              </button>
            </div>
            <div style={{ fontSize: '0.875rem', fontWeight: '500', color: '#fff' }}>
              {currentProvider?.name || currentConfig.provider}
            </div>
            <div style={{ fontSize: '0.7rem', color: '#aaa', marginTop: '0.125rem' }}>
              {currentConfig.model}
            </div>
            {currentProvider?.capabilities && Object.keys(currentProvider.capabilities).length > 0 && (
              <div style={{ fontSize: '0.65rem', color: '#888', marginTop: '0.25rem', display: 'flex', flexWrap: 'wrap', gap: '0.25rem' }}>
                {currentProvider.capabilities.tool_calling && (
                  <span style={{ background: '#2a3a2a', padding: '0.125rem 0.375rem', borderRadius: '3px' }}>
                    🔧 Tools
                  </span>
                )}
                {currentProvider.capabilities.image_inputs && (
                  <span style={{ background: '#2a2a3a', padding: '0.125rem 0.375rem', borderRadius: '3px' }}>
                    🖼️ Images
                  </span>
                )}
                {currentProvider.capabilities.reasoning_output && (
                  <span style={{ background: '#3a2a3a', padding: '0.125rem 0.375rem', borderRadius: '3px' }}>
                    🧠 Reasoning
                  </span>
                )}
                {currentProvider.capabilities.max_input_tokens && (
                  <span style={{ background: '#2a2a2a', padding: '0.125rem 0.375rem', borderRadius: '3px' }}>
                    📝 {Math.floor(currentProvider.capabilities.max_input_tokens / 1000)}k tokens
                  </span>
                )}
              </div>
            )}
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
              // Use strict comparison with trimmed and normalized strings
              const currentModel = (currentConfig.model || '').trim().toLowerCase();
              const providerModel = (provider.model || '').trim().toLowerCase();
              const isActive = currentModel === providerModel;
              
              const isUnhealthy = provider.health_status === 'unhealthy' || !provider.selectable;
              const isDisabled = loading || isActive || isUnhealthy;
              
              return (
                <button
                  key={provider.id}
                  onClick={() => !isDisabled && handleSwitchProvider(provider.id)}
                  disabled={isDisabled}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    padding: '0.625rem',
                    borderRadius: '4px',
                    border: isActive 
                      ? '1px solid #19c37d' 
                      : isUnhealthy 
                        ? '1px solid #6a2f2f' 
                        : '1px solid #444',
                    background: isActive 
                      ? '#1a3a2a' 
                      : isUnhealthy 
                        ? '#2a1a1a' 
                        : 'transparent',
                    color: isActive 
                      ? '#19c37d' 
                      : isUnhealthy 
                        ? '#ff6b6b' 
                        : '#fff',
                    fontSize: '0.8125rem',
                    cursor: isDisabled ? 'not-allowed' : 'pointer',
                    opacity: isDisabled ? 0.6 : 1,
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive && !loading && !isUnhealthy) {
                      e.currentTarget.style.background = '#2d2d30';
                      e.currentTarget.style.borderColor = '#555';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive && !loading && !isUnhealthy) {
                      e.currentTarget.style.background = 'transparent';
                      e.currentTarget.style.borderColor = '#444';
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.125rem' }}>
                    <div style={{ fontWeight: '500', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      {provider.name}
                      {isUnhealthy && (
                        <span style={{ fontSize: '0.75rem', color: '#ff6b6b' }} title={provider.error || 'Model unavailable'}>
                          ⚠️
                        </span>
                      )}
                    </div>
                    {provider.status && (
                      <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                        {provider.status}
                      </span>
                    )}
                  </div>
                  {provider.description && (
                    <div style={{ fontSize: '0.7rem', color: isUnhealthy ? '#ff9999' : '#aaa', marginTop: '0.125rem' }}>
                      {provider.description}
                    </div>
                  )}
                  {provider.capabilities && Object.keys(provider.capabilities).length > 0 ? (
                    <div style={{ fontSize: '0.65rem', color: '#888', marginTop: '0.375rem', display: 'flex', flexWrap: 'wrap', gap: '0.375rem' }}>
                      <div style={{ fontSize: '0.6rem', color: '#666', width: '100%', marginBottom: '0.125rem' }}>
                        Capabilities:
                      </div>
                      {provider.capabilities.tool_calling && (
                        <span style={{ background: '#2a3a2a', padding: '0.25rem 0.5rem', borderRadius: '4px', border: '1px solid #3a4a3a' }}>
                          🔧 Tools
                        </span>
                      )}
                      {provider.capabilities.image_inputs && (
                        <span style={{ background: '#2a2a3a', padding: '0.25rem 0.5rem', borderRadius: '4px', border: '1px solid #3a3a4a' }}>
                          🖼️ Images
                        </span>
                      )}
                      {provider.capabilities.reasoning_output && (
                        <span style={{ background: '#3a2a3a', padding: '0.25rem 0.5rem', borderRadius: '4px', border: '1px solid #4a3a4a' }}>
                          🧠 Reasoning
                        </span>
                      )}
                      {provider.capabilities.max_input_tokens && (
                        <span style={{ background: '#2a2a2a', padding: '0.25rem 0.5rem', borderRadius: '4px', border: '1px solid #3a3a3a' }}>
                          📝 {Math.floor(provider.capabilities.max_input_tokens / 1000)}k tokens
                        </span>
                      )}
                    </div>
                  ) : (
                    <div style={{ fontSize: '0.65rem', color: '#666', marginTop: '0.375rem', fontStyle: 'italic' }}>
                      No capabilities information available
                    </div>
                  )}
                  {provider.error && (
                    <div style={{ fontSize: '0.65rem', color: '#ff6b6b', marginTop: '0.25rem', fontStyle: 'italic' }}>
                      {provider.error}
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

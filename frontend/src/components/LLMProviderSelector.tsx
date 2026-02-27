/**Router Status — shows LLM router tiers and their availability.
 *
 * Replaces the old per-model provider selector.
 * Model selection is now automatic (router handles it per session).
 * This component is read-only: shows which models are available per tier.
 */

import { useEffect, useState } from 'react';
import * as api from '../api/client';
import type { RouterTierStatus, RouterConfig } from '../types/api';

const TIER_ORDER = ['nano', 'fast', 'smart', 'max'];

const TIER_ICONS: Record<string, string> = {
  nano: '💬',
  fast: '⚡',
  smart: '🧠',
  max: '🚀',
};

interface TierInfo extends RouterTierStatus {
  description?: string;
  config_model?: string;
}

export function LLMProviderSelector() {
  const [routerConfig, setRouterConfig] = useState<RouterConfig | null>(null);
  const [tierStatus, setTierStatus] = useState<Record<string, TierInfo>>({});
  const [expanded, setExpanded] = useState(false);
  const [restarting, setRestarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    setLoading(true);
    try {
      const [cfgRes, tiersRes] = await Promise.all([
        api.getRouterConfig(),
        api.getModelTiers(),
      ]);

      setRouterConfig(cfgRes.router);

      // Merge tier config (descriptions) with runtime status
      const merged: Record<string, TierInfo> = {};
      const cfgTiers = cfgRes.router?.tiers ?? {};
      const statusTiers = tiersRes.tiers ?? {};

      const allTierNames = new Set([...Object.keys(cfgTiers), ...Object.keys(statusTiers)]);
      for (const name of allTierNames) {
        merged[name] = {
          selected_model: statusTiers[name]?.selected_model ?? null,
          preferred_models: statusTiers[name]?.preferred_models ?? cfgTiers[name]?.preferred_models ?? [],
          available: statusTiers[name]?.available ?? [],
          missing: statusTiers[name]?.missing ?? [],
          pull_command: statusTiers[name]?.pull_command ?? null,
          description: cfgTiers[name]?.description,
          config_model: cfgTiers[name]?.model,
        };
      }
      setTierStatus(merged);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load router config');
    } finally {
      setLoading(false);
    }
  }

  async function handleRestartOllama() {
    setRestarting(true);
    setError(null);
    setSuccess(null);
    try {
      await api.restartOllama();
      setSuccess('Ollama reiniciado com sucesso');
      await loadData();
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restart Ollama');
      setTimeout(() => setError(null), 5000);
    } finally {
      setRestarting(false);
    }
  }

  const defaultTier = routerConfig?.default_tier ?? 'smart';

  // Summary line: "nano · fast · smart✓ · max"
  const tierSummary = TIER_ORDER.filter((t) => tierStatus[t]).map((t) => {
    const info = tierStatus[t];
    const ok = !!info.selected_model;
    const label = t === defaultTier ? `${t}✓` : t;
    return ok ? label : `${label}✗`;
  }).join(' · ');

  return (
    <div>
      {/* Collapsed header */}
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
        onMouseEnter={(e) => { if (!expanded) e.currentTarget.style.background = '#2d2d30'; }}
        onMouseLeave={(e) => { if (!expanded) e.currentTarget.style.background = 'transparent'; }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
              <div style={{ fontSize: '0.75rem', color: '#888' }}>LLM Router</div>
              <button
                onClick={(e) => { e.stopPropagation(); handleRestartOllama(); }}
                disabled={restarting}
                title="Reiniciar Ollama"
                style={{
                  background: 'transparent', border: 'none',
                  color: restarting ? '#888' : '#19c37d',
                  cursor: restarting ? 'not-allowed' : 'pointer',
                  padding: '0.125rem 0.375rem',
                  fontSize: '0.875rem',
                  opacity: restarting ? 0.5 : 1,
                  borderRadius: '4px',
                }}
                onMouseEnter={(e) => { if (!restarting) { e.currentTarget.style.background = '#1a3a2a'; e.currentTarget.style.color = '#4ade80'; } }}
                onMouseLeave={(e) => { if (!restarting) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = '#19c37d'; } }}
              >
                {restarting ? '⟳' : '↻'}
              </button>
            </div>
            {loading ? (
              <div style={{ fontSize: '0.75rem', color: '#888' }}>Carregando...</div>
            ) : (
              <>
                <div style={{ fontSize: '0.8rem', fontWeight: '500', color: '#fff' }}>
                  Roteamento automático
                </div>
                <div style={{ fontSize: '0.65rem', color: '#888', marginTop: '0.125rem', fontFamily: 'monospace' }}>
                  {tierSummary}
                </div>
              </>
            )}
          </div>
          <div style={{ color: '#888', fontSize: '0.75rem' }}>{expanded ? '▼' : '▶'}</div>
        </div>
      </div>

      {/* Expanded tier list */}
      {expanded && (
        <div style={{ marginTop: '0.5rem', padding: '0.5rem', background: '#1a1a1c', borderRadius: '6px', border: '1px solid #333' }}>
          {error && (
            <div style={{ padding: '0.5rem', background: '#4a1f1f', border: '1px solid #6a2f2f', borderRadius: '4px', color: '#ff6b6b', fontSize: '0.75rem', marginBottom: '0.5rem' }}>
              {error}
            </div>
          )}
          {success && (
            <div style={{ padding: '0.5rem', background: '#1f4a1f', border: '1px solid #2f6a2f', borderRadius: '4px', color: '#6bff6b', fontSize: '0.75rem', marginBottom: '0.5rem' }}>
              {success}
            </div>
          )}

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.375rem' }}>
            {TIER_ORDER.filter((t) => tierStatus[t]).map((tierName) => {
              const info = tierStatus[tierName];
              const available = !!info.selected_model;
              const isDefault = tierName === defaultTier;

              return (
                <div
                  key={tierName}
                  style={{
                    padding: '0.625rem',
                    borderRadius: '4px',
                    border: available
                      ? isDefault ? '1px solid #19c37d' : '1px solid #444'
                      : '1px solid #6a2f2f',
                    background: available
                      ? isDefault ? '#1a3a2a' : 'transparent'
                      : '#2a1a1a',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <span style={{ fontSize: '0.875rem' }}>{TIER_ICONS[tierName] ?? '•'}</span>
                      <span style={{
                        fontSize: '0.8125rem', fontWeight: '600',
                        color: available ? (isDefault ? '#19c37d' : '#fff') : '#ff6b6b',
                        textTransform: 'uppercase', letterSpacing: '0.05em',
                      }}>
                        {tierName}
                      </span>
                      {isDefault && (
                        <span style={{ fontSize: '0.6rem', color: '#19c37d', background: '#1a3a2a', border: '1px solid #19c37d', borderRadius: '3px', padding: '0.1rem 0.3rem' }}>
                          padrão
                        </span>
                      )}
                    </div>
                    <span style={{ fontSize: '0.75rem' }}>{available ? '✅' : '❌'}</span>
                  </div>

                  <div style={{ fontSize: '0.7rem', color: '#aaa', marginBottom: '0.25rem' }}>
                    {info.description ?? ''}
                  </div>

                  <div style={{ fontSize: '0.7rem', fontFamily: 'monospace', color: available ? '#4ade80' : '#ff9999' }}>
                    {info.selected_model ?? info.config_model ?? info.preferred_models[0] ?? '—'}
                  </div>

                  {!available && info.pull_command && (
                    <div style={{ marginTop: '0.375rem', fontSize: '0.65rem', color: '#888' }}>
                      <span style={{ color: '#666' }}>Instalar: </span>
                      <code style={{ background: '#111', padding: '0.1rem 0.3rem', borderRadius: '3px', color: '#fbbf24' }}>
                        {info.pull_command}
                      </code>
                    </div>
                  )}

                  {available && info.missing.length > 0 && (
                    <div style={{ marginTop: '0.25rem', fontSize: '0.6rem', color: '#666' }}>
                      Preferidos indisponíveis: {info.missing.join(', ')}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: '0.5rem', fontSize: '0.65rem', color: '#555', textAlign: 'center' }}>
            Seleção automática por mensagem · padrão: {defaultTier}
          </div>
        </div>
      )}
    </div>
  );
}

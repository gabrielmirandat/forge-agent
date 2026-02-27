"""LLM Router — rule-based automatic model selection.

Reads config/router.yaml and selects the appropriate model tier
(fast / smart / max) based on keyword matching in the user message.

The selected model is applied per-session: once a session has an executor,
the model is fixed for the lifetime of that session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    """Result of routing a user message to a model tier."""

    model: str
    tier: str
    reason: str


class LLMRouter:
    """Rule-based LLM router.

    Reads router.yaml and matches keywords in user messages to route
    to the appropriate model tier.
    """

    def __init__(self, config_path: str = "config/router.yaml") -> None:
        self._config_path = Path(config_path)
        self._cfg: Dict[str, Any] = {}
        self._enabled: bool = False
        self._load()

    def _load(self) -> None:
        if not self._config_path.exists():
            logger.warning(f"Router config not found at {self._config_path} — routing disabled")
            return
        with open(self._config_path) as f:
            self._cfg = yaml.safe_load(f) or {}
        self._enabled = self._cfg.get("router", {}).get("enabled", False)
        if self._enabled:
            tiers = self._cfg.get("router", {}).get("tiers", {})
            logger.info(f"LLM Router loaded: enabled={self._enabled}, tiers={list(tiers.keys())}")

    def reload(self) -> None:
        """Reload config from disk."""
        self._load()

    def update_tier_model(self, tier_name: str, model: str) -> None:
        """Update the resolved model for a tier (called after startup discovery)."""
        tiers = self._cfg.get("router", {}).get("tiers", {})
        if tier_name in tiers:
            tiers[tier_name]["model"] = model
            logger.info(f"Router tier '{tier_name}' updated to model '{model}'")

    def route(self, message: str) -> RouterDecision:
        """Route a message to the appropriate model tier.

        Args:
            message: User message content

        Returns:
            RouterDecision with model name, tier, and reason
        """
        router_cfg = self._cfg.get("router", {})

        # If router disabled or no config, use default tier
        if not self._enabled or not router_cfg:
            return self._build_decision(router_cfg.get("default_tier", "smart"), "router disabled")

        msg_lower = message.lower()

        for rule in router_cfg.get("rules", []):
            tier_name = rule.get("tier", "")
            keywords = rule.get("keywords", [])
            if any(kw in msg_lower for kw in keywords):
                return self._build_decision(tier_name, "keyword match")

        default_tier = router_cfg.get("default_tier", "smart")
        return self._build_decision(default_tier, "default")

    def _build_decision(self, tier_name: str, reason: str) -> RouterDecision:
        tiers = self._cfg.get("router", {}).get("tiers", {})
        if tier_name not in tiers:
            # Fallback: return first available tier or a safe default
            logger.warning(f"Router: unknown tier '{tier_name}', falling back to first available")
            tier_name = next(iter(tiers), "smart")
            if tier_name not in tiers:
                return RouterDecision(model="qwen3:14b", tier="smart", reason="no tiers configured")

        tier_cfg = tiers[tier_name]
        model = tier_cfg.get("model", "qwen3:14b")
        return RouterDecision(model=model, tier=tier_name, reason=reason)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def config(self) -> Dict[str, Any]:
        return self._cfg.get("router", {})


# Module-level singleton (lazy-loaded)
_router: Optional[LLMRouter] = None


def get_router(config_path: str = "config/router.yaml") -> LLMRouter:
    """Get global LLM router instance."""
    global _router
    if _router is None:
        _router = LLMRouter(config_path)
    return _router

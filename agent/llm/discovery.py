"""Dynamic model discovery via Ollama API.

Discovers locally available models without relying on YAML config files.
Also provides tier resolution: given a list of preferred models per tier,
selects the best available one.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)


async def discover_ollama_models(base_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:
    """Discover models available in Ollama.

    Merges /api/tags (all downloaded models) with /api/ps (models currently in VRAM).

    Returns:
        List of dicts with keys: name, size_gb, in_vram, digest
    """
    tags_data: List[Dict] = []
    ps_names: Set[str] = set()

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                tags_data = resp.json().get("models", [])
        except Exception as e:
            logger.warning(f"discover_ollama_models: /api/tags failed: {e}")

        try:
            resp = await client.get(f"{base_url}/api/ps")
            if resp.status_code == 200:
                for m in resp.json().get("models", []):
                    name = m.get("name", "")
                    if name:
                        ps_names.add(name)
        except Exception as e:
            logger.debug(f"discover_ollama_models: /api/ps failed (non-critical): {e}")

    result = []
    for m in tags_data:
        name = m.get("name", "")
        if not name:
            continue
        size_bytes = m.get("size", 0) or 0
        result.append({
            "name": name,
            "size_gb": round(size_bytes / 1e9, 1),
            "in_vram": name in ps_names,
            "digest": m.get("digest", ""),
        })

    return result


async def resolve_tier_models(
    router_config: Dict[str, Any],
    ollama_url: str = "http://localhost:11434",
) -> Dict[str, Dict[str, Any]]:
    """For each router tier, select the best locally available model.

    Args:
        router_config: The "router" section from router.yaml
        ollama_url: Ollama API base URL

    Returns:
        Dict mapping tier_name -> {selected_model, preferred_models, available, missing, pull_command}
    """
    available_models = await discover_ollama_models(ollama_url)
    available_names: Set[str] = {m["name"] for m in available_models}

    result: Dict[str, Dict[str, Any]] = {}
    tiers = router_config.get("tiers", {})

    for tier_name, tier_cfg in tiers.items():
        preferred = tier_cfg.get("preferred_models", [])
        # If no preferred_models list, fall back to the single model field
        if not preferred and tier_cfg.get("model"):
            preferred = [tier_cfg["model"]]

        available_in_tier = [m for m in preferred if m in available_names]
        missing_in_tier = [m for m in preferred if m not in available_names]
        selected = available_in_tier[0] if available_in_tier else None

        result[tier_name] = {
            "selected_model": selected,
            "preferred_models": preferred,
            "available": available_in_tier,
            "missing": missing_in_tier,
            "status": "available" if selected else "missing",
            "pull_command": f"ollama pull {preferred[0]}" if not selected and preferred else None,
        }

    return result

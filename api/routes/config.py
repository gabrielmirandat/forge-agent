"""Configuration API routes for runtime LLM provider switching."""

from agent.id import ascending
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

from agent.config.loader import AgentConfig, LLMConfig
from agent.observability import get_logger, log_event, set_request_id
from agent.runtime.bus import EventType, publish
from api.dependencies import get_config

router = APIRouter()
logger = get_logger("api.config", "api")


@router.get("/config/llm")
async def get_llm_config(config: AgentConfig = Depends(get_config)) -> Dict[str, Any]:
    """Get current LLM configuration.
    
    Returns:
        Current LLM configuration
    """
    request_id = ascending("message")  # Request ID for correlation
    set_request_id(request_id)
    
    llm_config = config.llm.model_dump()
    return {
        "provider": llm_config.get("provider"),
        "model": llm_config.get("model"),
        "temperature": llm_config.get("temperature"),
        "max_tokens": llm_config.get("max_tokens"),
        "timeout": llm_config.get("timeout"),
        "base_url": llm_config.get("base_url"),
        "compression": llm_config.get("compression"),
        "profiling_mode": llm_config.get("profiling_mode"),
        "layer_shards_saving_path": llm_config.get("layer_shards_saving_path"),
        "delete_original": llm_config.get("delete_original"),
        "hf_token": None,  # Never expose token
    }


@router.get("/config/llm/providers")
async def list_llm_providers() -> Dict[str, Any]:
    """List available LLM providers and models by scanning config directory.
    
    Automatically discovers providers from agent.*.yaml files in config/ directory.
    Only includes models that have been tested and verified to work with tool calling.
    
    Returns:
        Available providers and models discovered from config files
    """
    from pathlib import Path
    import yaml
    
    config_dir = Path("config")
    providers = []
    
    # Models that have been tested and verified to work with MCP tool calling
    # ✅ = Works correctly, ❌ = Does not work (model limitation)
    tested_models = {
        "qwen3:8b": {"status": "✅", "description": "Qwen model with good tool-calling support"},
        # "mistral": {"status": "❌", "description": "Does not call tools (model limitation)"},  # Removed - not working
    }
    
    # Track seen models to avoid duplicates (prefer specific configs over generic ones)
    seen_models = set()
    
    # First pass: collect specific model configs (e.g., agent.ollama.llama31.yaml)
    specific_configs = []
    generic_configs = []
    
    for config_file in config_dir.glob("agent.*.yaml"):
        # Skip agent.yaml (default/main config)
        if config_file.name == "agent.yaml":
            continue
        
        try:
            # Extract provider name from filename
            provider_id = config_file.stem.replace("agent.", "")
            
            # Load config to get provider details
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)
            
            if not config_data or "agent" not in config_data:
                continue
            
            llm_config = config_data.get("agent", {}).get("llm", {})
            provider_name = llm_config.get("provider", provider_id.split(".")[0])
            model = llm_config.get("model", "unknown")
            
            # Categorize: specific (has model in filename) vs generic
            if len(provider_id.split(".")) > 2:  # e.g., agent.ollama.llama31.yaml
                specific_configs.append((config_file, provider_id, provider_name, model))
            else:
                generic_configs.append((config_file, provider_id, provider_name, model))
        except Exception as e:
            logger.warning(f"Failed to parse config file {config_file}: {e}")
            continue
    
    # Process specific configs first (they take priority)
    for config_file, provider_id, provider_name, model in specific_configs:
        if model in seen_models:
            continue  # Skip if we already have this model from a specific config
        
        model_info = tested_models.get(model, {})
        model_status = model_info.get("status", "⚠️")
        model_description = model_info.get("description", f"{model} model")
        
        descriptions = {
            "ollama": "Local LLM via Docker (recommended for easy setup)",
        }
        
        provider_description = descriptions.get(provider_name, f"{provider_name.capitalize()} provider")
        if model_status == "✅":
            provider_description += f" - {model_description}"
        
        providers.append({
            "id": provider_id,
            "name": f"{provider_name.capitalize()} ({model})",
            "description": provider_description,
            "config_file": str(config_file),
            "model": model,
            "status": model_status,
        })
        seen_models.add(model)
    
    # Process generic configs (only if model not already seen)
    for config_file, provider_id, provider_name, model in generic_configs:
        if model in seen_models:
            continue  # Skip if we already have this model
        
        model_info = tested_models.get(model, {})
        model_status = model_info.get("status", "⚠️")
        model_description = model_info.get("description", f"{model} model")
        
        descriptions = {
            "ollama": "Local LLM via Docker (recommended for easy setup)",
        }
        
        provider_description = descriptions.get(provider_name, f"{provider_name.capitalize()} provider")
        if model_status == "✅":
            provider_description += f" - {model_description}"
        
        providers.append({
            "id": provider_id,
            "name": f"{provider_name.capitalize()} ({model})",
            "description": provider_description,
            "config_file": str(config_file),
            "model": model,
            "status": model_status,
        })
        seen_models.add(model)
    
    # Sort providers: ✅ first, then others
    providers.sort(key=lambda p: (p.get("status", "⚠️") != "✅", p.get("name", "")))
    
    return {"providers": providers}


@router.put("/config/llm")
async def update_llm_config(
    llm_config: Dict[str, Any],
    config: AgentConfig = Depends(get_config),
) -> Dict[str, Any]:
    """Update LLM configuration at runtime.
    
    This switches the LLM provider without restarting the server.
    The new provider will be used for all subsequent requests.
    
    Args:
        llm_config: New LLM configuration
        config: Current agent configuration (injected)
    
    Returns:
        Updated LLM configuration
    """
    request_id = ascending("message")  # Request ID for correlation
    set_request_id(request_id)
    
    log_event(
        logger,
        "api.config.llm.update",
        provider=llm_config.get("provider"),
        model=llm_config.get("model"),
    )
    
    try:
        # Validate new LLM config
        new_llm_config = LLMConfig(**llm_config)
        
        # Update config cache
        import api.dependencies as deps
        if deps._config_cache is None:
            deps._config_cache = config
        deps._config_cache.llm = new_llm_config
        
        # Clear LLM provider cache to force reload
        deps._llm_provider_cache = None
        
        # Publish event
        await publish(EventType.SESSION_UPDATED, {
            "type": "llm_provider_changed",
            "provider": new_llm_config.provider,
            "model": new_llm_config.model,
        })
        
        logger.info(
            f"LLM provider updated: {new_llm_config.provider} / {new_llm_config.model}",
            request_id=request_id,
        )
        
        return {
            "status": "success",
            "message": f"LLM provider switched to {new_llm_config.provider}",
            "config": {
                "provider": new_llm_config.provider,
                "model": new_llm_config.model,
                "temperature": new_llm_config.temperature,
                "max_tokens": new_llm_config.max_tokens,
                "timeout": new_llm_config.timeout,
            },
        }
        
    except Exception as e:
        logger.error(f"Failed to update LLM config: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid LLM configuration: {str(e)}",
        ) from e


@router.post("/config/llm/switch")
async def switch_llm_provider(
    provider: str,
    config_file: str | None = None,
    config: AgentConfig = Depends(get_config),
) -> Dict[str, Any]:
    """Switch LLM provider by loading from config file.
    
    This is a convenience endpoint that loads a pre-configured provider
    from agent.ollama.yaml.
    
    Args:
        provider: Provider name ("ollama")
        config_file: Optional path to config file (defaults to agent.{provider}.yaml)
        config: Current agent configuration (injected)
    
    Returns:
        Updated LLM configuration
    """
    request_id = ascending("message")  # Request ID for correlation
    set_request_id(request_id)

    # Provider may include model identifier (e.g., "ollama.llama31" or "ollama.qwen")
    # We treat the first segment as the base provider ("ollama") and the full value
    # as the provider_id used in the config filename.
    parts = provider.split(".")
    base_provider = parts[0] if parts else provider
    provider_id = provider  # Used to resolve config file

    if base_provider != "ollama":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid provider: {provider}. Only 'ollama' is supported.",
        )
    
    # Determine config file path
    if config_file is None:
        # agent.ollama.yaml, agent.ollama.llama31.yaml, agent.ollama.qwen.yaml, etc.
        config_file = f"config/agent.{provider_id}.yaml"
    
    from pathlib import Path
    config_path = Path(config_file).expanduser().resolve()
    
    if not config_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Config file not found: {config_file}",
        )
    
    try:
        # Load config from file
        from agent.config.loader import ConfigLoader
        loader = ConfigLoader(config_path=str(config_path))
        new_config = loader.load()
        
        # Update config cache
        import api.dependencies as deps
        if deps._config_cache is None:
            deps._config_cache = config
        deps._config_cache.llm = new_config.llm
        
        # Clear LLM provider cache
        deps._llm_provider_cache = None
        
        # Publish event
        await publish(EventType.SESSION_UPDATED, {
            "type": "llm_provider_changed",
            "provider": new_config.llm.provider,
            "model": new_config.llm.model,
            "config_file": str(config_file),
        })
        
        log_event(
            logger,
            "llm.provider.switched",
            provider=provider,
            model=new_config.llm.model,
            config_file=str(config_file),
            request_id=request_id,
        )
        
        return {
            "status": "success",
            "message": f"Switched to {provider} provider",
            "config": {
                "provider": new_config.llm.provider,
                "model": new_config.llm.model,
                "temperature": new_config.llm.temperature,
                "max_tokens": new_config.llm.max_tokens,
                "timeout": new_config.llm.timeout,
            },
        }
        
    except Exception as e:
        logger.error(f"Failed to switch LLM provider: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to switch provider: {str(e)}",
        ) from e

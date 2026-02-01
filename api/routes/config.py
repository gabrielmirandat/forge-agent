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
    """List available LLM providers and models with health status and capabilities.
    
    Uses ModelManager to get pre-initialized models with health checks and capabilities.
    
    Returns:
        Available providers with health status, capabilities, and model instances
    """
    from agent.llm.model_manager import get_model_manager
    
    manager = get_model_manager()
    all_models = manager.get_all_models()
    
    providers = []
    
    for provider_id, model_instance in all_models.items():
        # Determine status emoji based on health
        if model_instance.health_status == "healthy":
            status_emoji = "✅"
        elif model_instance.health_status == "unhealthy":
            status_emoji = "❌"
        else:
            status_emoji = "⚠️"
        
        # Extract provider name from ID
        provider_name = provider_id.split(".")[0] if "." in provider_id else provider_id
        
        # Build description with capabilities
        description_parts = []
        if model_instance.capabilities:
            caps = []
            if model_instance.capabilities.get("tool_calling"):
                caps.append("tool calling")
            if model_instance.capabilities.get("image_inputs"):
                caps.append("image inputs")
            if model_instance.capabilities.get("reasoning_output"):
                caps.append("reasoning")
            if model_instance.capabilities.get("max_input_tokens"):
                max_tokens = model_instance.capabilities.get("max_input_tokens")
                caps.append(f"{max_tokens:,} max tokens")
            
            if caps:
                description_parts.append(f"Capabilities: {', '.join(caps)}")
        
        if model_instance.error:
            description_parts.append(f"Error: {model_instance.error}")
        
        description = " | ".join(description_parts) if description_parts else "Local LLM provider"
        
        providers.append({
            "id": provider_id,
            "name": f"{provider_name.capitalize()} ({model_instance.model_name})",
            "description": description,
            "config_file": None,  # Not needed when using ModelManager
            "model": model_instance.model_name,
            "status": status_emoji,
            "health_status": model_instance.health_status,
            "capabilities": model_instance.capabilities,
            "error": model_instance.error,
            "selectable": model_instance.health_status == "healthy",
        })
    
    # Sort providers: healthy first, then others
    providers.sort(
        key=lambda p: (
            p.get("health_status") != "healthy",
            p.get("name", ""),
        )
    )
    
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
        
        # Clear LangChain executor cache to force recreation with new model
        from agent.runtime.langchain_executor import clear_shared_executor
        clear_shared_executor()
        
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
    from config files like agent.ollama.qwen.yaml, agent.ollama.mistral.yaml, etc.
    
    Args:
        provider: Provider name with model identifier (e.g., "ollama.qwen", "ollama.mistral")
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
        # agent.ollama.qwen.yaml, agent.ollama.mistral.yaml, agent.ollama.deepseek.yaml, etc.
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
        
        # Clear LangChain executor cache to force recreation with new model
        from agent.runtime.langchain_executor import clear_shared_executor
        clear_shared_executor()
        
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


@router.post("/config/llm/restart")
async def restart_ollama(
    config: AgentConfig = Depends(get_config),
) -> Dict[str, Any]:
    """Restart Ollama Docker container.
    
    This endpoint restarts the Ollama container to refresh the model state.
    Useful when experiencing delays or issues with model responses.
    
    Args:
        config: Current agent configuration (injected)
    
    Returns:
        Status of restart operation
    """
    request_id = ascending("message")
    set_request_id(request_id)
    
    if config.llm.provider.lower() != "ollama":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ollama restart is only available when provider is 'ollama'",
        )
    
    try:
        from agent.runtime.docker_manager import get_docker_manager
        
        docker_manager = get_docker_manager()
        
        # Extract port from base_url
        base_url = config.llm.base_url or "http://localhost:11434"
        try:
            port = int(base_url.split(":")[-1]) if ":" in base_url else 11434
        except ValueError:
            port = 11434
        
        # Restart Ollama container
        success = await docker_manager.restart_ollama(port=port, gpu=True)
        
        if success:
            # Re-initialize model manager after restart
            from agent.llm.model_manager import get_model_manager, ModelManager
            from pathlib import Path
            
            # Clear existing models and re-initialize
            manager = get_model_manager()
            manager.models.clear()
            manager._initialized = False
            
            config_dir = Path("config")
            await manager.initialize_all_models(config_dir)
            
            log_event(
                logger,
                "llm.ollama.restarted",
                port=port,
                request_id=request_id,
            )
            
            return {
                "status": "success",
                "message": "Ollama container restarted successfully",
                "port": port,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to restart Ollama container",
            )
            
    except Exception as e:
        logger.error(f"Failed to restart Ollama: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to restart Ollama: {str(e)}",
        ) from e

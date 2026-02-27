"""FastAPI main application."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from prometheus_client import make_asgi_app

# Load environment variables from .env file
# Look for .env in the project root (parent of api/)
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print(f"No .env file found at {env_path} (using system environment variables)")

from api.dependencies import get_config, initialize_tool_registry
from api.routes import config, events, health, observability, session

app = FastAPI(
    title="Forge Agent API",
    description="API for self-hosted autonomous code agent",
    version="0.1.0",
)

# Register routes
app.include_router(health.router, tags=["health"])
app.include_router(events.router, prefix="/api/v1", tags=["events"])
app.include_router(config.router, prefix="/api/v1", tags=["config"])
app.include_router(session.router, prefix="/api/v1", tags=["sessions"])
app.include_router(observability.router, prefix="/api/v1", tags=["observability"])

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.on_event("startup")
async def startup_event():
    """Initialize Docker containers and tool registry on startup."""
    config = get_config()
    
    from agent.runtime.docker_manager import get_docker_manager
    
    docker_manager = get_docker_manager()
    
    # Start Ollama Docker container if provider is ollama
    # This ensures Ollama is running before LangChainExecutor tries to connect
    if config.llm.provider.lower() == "ollama":
        # Extract port from base_url if present, default to 11434
        base_url = config.llm.base_url or "http://localhost:11434"
        try:
            port = int(base_url.split(":")[-1]) if ":" in base_url else 11434
        except ValueError:
            port = 11434
        
        # Start Ollama Docker container (waits for health check)
        ollama_started = await docker_manager.start_ollama(port=port, gpu=True)
        if ollama_started:
            print(f"✅ Ollama Docker container started and ready on port {port}")
            print(f"   LangChain will connect to: {base_url}")
        else:
            print(f"❌ Failed to start Ollama Docker container")
            print(f"   LangChain may fail to connect. Check Docker logs.")
    
    # Ensure Docker images are available for MCP servers
    mcp_configs = config.mcp or {}
    image_results = await docker_manager.ensure_images(mcp_configs)
    
    # Log results
    for mcp_name, success in image_results.items():
        if success:
            print(f"✅ Docker image available for MCP server: {mcp_name}")
        else:
            print(f"❌ Failed to ensure Docker image for MCP server: {mcp_name}")
    
    # Initialize model manager - pre-initialize all models with health checks
    from agent.llm.model_manager import initialize_model_manager, discover_and_register_models
    from pathlib import Path

    config_dir = Path("config")
    print("🔄 Pre-initializing all available models...")
    model_manager = await initialize_model_manager(config_dir)

    # Also discover models via Ollama API (catches models not in YAML files)
    ollama_url = config.llm.base_url or "http://localhost:11434"
    await discover_and_register_models(ollama_url)

    # Log model initialization results
    all_models = model_manager.get_all_models()
    healthy_models = model_manager.get_healthy_models()
    print(f"✅ Model manager initialized: {len(healthy_models)}/{len(all_models)} models healthy")
    for provider_id, model_instance in all_models.items():
        status_icon = "✅" if model_instance.health_status == "healthy" else "❌"
        print(f"   {status_icon} {model_instance.model_name} ({provider_id}): {model_instance.health_status}")
        if model_instance.capabilities:
            caps = []
            if model_instance.capabilities.get("tool_calling"):
                caps.append("tools")
            if model_instance.capabilities.get("image_inputs"):
                caps.append("images")
            if model_instance.capabilities.get("reasoning_output"):
                caps.append("reasoning")
            if caps:
                print(f"      Capabilities: {', '.join(caps)}")

    # Resolve router tiers against locally available models
    from agent.routing.router import get_router
    from agent.llm.discovery import resolve_tier_models

    rtr = get_router()
    if rtr.enabled:
        print("🔀 Resolving LLM router tiers...")
        try:
            tier_results = await resolve_tier_models(rtr.config, ollama_url)
            for tier_name, info in tier_results.items():
                if info["selected_model"]:
                    rtr.update_tier_model(tier_name, info["selected_model"])
                    print(f"   ✅ Tier '{tier_name}': {info['selected_model']}")
                else:
                    preferred = info.get("preferred_models", [])
                    pull_cmd = f"ollama pull {preferred[0]}" if preferred else "N/A"
                    print(f"   ⚠️  Tier '{tier_name}': no model available — {pull_cmd}")
        except Exception as e:
            print(f"   ⚠️  Router tier resolution failed: {e}")
    
    # Initialize tool registry
    await initialize_tool_registry(config)
    
    # Load tools and build system prompt for debug
    from api.dependencies import get_tool_registry
    from agent.runtime.langchain_executor import LangChainExecutor
    
    tool_registry = get_tool_registry()
    
    # Load LangChain tools to build the actual system prompt
    langchain_tools = await tool_registry.get_langchain_tools(
        session_id=None,
        config=config,
    )
    
    # Format system prompt using the static method (no need to create executor instance)
    system_prompt_str = await LangChainExecutor.format_system_prompt(config, langchain_tools)
    
    # Print system prompt for debug
    print("\n" + "="*80)
    print("FORGE AGENT STARTUP COMPLETE")
    print("="*80)
    print(f"✅ Workspace: {config.workspace.base_path}")
    print(f"✅ LLM Provider: {config.llm.provider} ({config.llm.model})")
    print("\n" + "="*80)
    print("SYSTEM PROMPT (what will be sent to LLM):")
    print("="*80)
    print(system_prompt_str)
    print("="*80 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup Docker manager and MCP connections on shutdown."""
    from agent.runtime.docker_manager import get_docker_manager
    from agent.runtime.mcp_client import get_mcp_manager
    
    try:
        # Disconnect all MCP servers first
        mcp_manager = get_mcp_manager()
        await mcp_manager.disconnect_all()
        print("🛑 MCP servers disconnected")
    except Exception as e:
        print(f"⚠️ Error disconnecting MCP servers: {e}")
    
    try:
        # Stop Docker containers
        docker_manager = get_docker_manager()
        await docker_manager.stop_containers()
        print("🛑 Docker manager stopped")
    except Exception as e:
        print(f"⚠️ Error stopping Docker containers: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Forge Agent API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "sessions": "/api/v1/sessions",
            "events": "/api/v1/events",
            "config": "/api/v1/config",
        },
    }


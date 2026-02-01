"""Model Manager - Pre-initializes all available models and manages health checks.

This module pre-initializes all available LLM models at application startup,
performs health checks, and retrieves capabilities using model.profile.
Models are stored in a dictionary for fast switching without re-initialization.
"""

import asyncio
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml
import httpx
import logging

from agent.config.loader import AgentConfig, LLMConfig
from agent.llm.ollama import OllamaProvider

logger = logging.getLogger(__name__)


class ModelInstance:
    """Represents an initialized model instance with health status and capabilities."""
    
    def __init__(
        self,
        model_name: str,
        provider_id: str,
        config: Dict[str, Any],
        langchain_model: Any = None,
        health_status: str = "unknown",  # "healthy", "unhealthy", "unknown"
        capabilities: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.model_name = model_name
        self.provider_id = provider_id
        self.config = config
        self.langchain_model = langchain_model
        self.health_status = health_status
        self.capabilities = capabilities or {}
        self.error = error


class ModelManager:
    """Manages pre-initialized model instances for fast switching."""
    
    def __init__(self):
        """Initialize model manager."""
        self.models: Dict[str, ModelInstance] = {}  # key: provider_id, value: ModelInstance
        self._initialized = False
    
    async def initialize_all_models(self, config_dir: Path = Path("config")) -> None:
        """Pre-initialize all available models from config files.
        
        Args:
            config_dir: Directory containing agent.*.yaml config files
        """
        if self._initialized:
            logger.info("Model manager already initialized")
            return
        
        logger.info("Starting pre-initialization of all models...")
        
        # Discover all config files
        config_files = list(config_dir.glob("agent.*.yaml"))
        if config_dir.name == "agent.yaml":
            config_files = []
        
        # Filter out main config
        config_files = [f for f in config_files if f.name != "agent.yaml"]
        
        if not config_files:
            logger.warning(f"No model config files found in {config_dir}")
            self._initialized = True
            return
        
        # Initialize all models in parallel
        tasks = []
        for config_file in config_files:
            tasks.append(self._initialize_model_from_config(config_file))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful initializations
        successful = sum(1 for r in results if isinstance(r, ModelInstance))
        failed = len(results) - successful
        
        logger.info(
            f"Model initialization complete: {successful} successful, {failed} failed"
        )
        
        self._initialized = True
    
    async def _initialize_model_from_config(
        self, config_file: Path
    ) -> Optional[ModelInstance]:
        """Initialize a single model from config file.
        
        Args:
            config_file: Path to agent.*.yaml config file
            
        Returns:
            ModelInstance or None if initialization failed
        """
        try:
            # Extract provider ID from filename
            provider_id = config_file.stem.replace("agent.", "")
            
            # Load config
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)
            
            if not config_data or "agent" not in config_data:
                logger.warning(f"Invalid config file: {config_file}")
                return None
            
            llm_config = config_data.get("agent", {}).get("llm", {})
            model_name = llm_config.get("model", "unknown")
            base_url = llm_config.get("base_url", "http://localhost:11434")
            
            # Create OllamaProvider config
            provider_config = {
                "model": model_name,
                "base_url": base_url,
                "temperature": llm_config.get("temperature", 0.0),
                "max_tokens": llm_config.get("max_tokens", 4096),
                "timeout": llm_config.get("timeout", 300),
            }
            
            # Initialize LangChain ChatOllama model
            langchain_model = None
            health_status = "unknown"
            capabilities = {}
            error = None
            
            try:
                # Try to import ChatOllama
                try:
                    from langchain_ollama import ChatOllama
                except ImportError:
                    from langchain_community.chat_models import ChatOllama
                
                # Create ChatOllama instance
                langchain_model = ChatOllama(
                    model=model_name,
                    base_url=base_url,
                    temperature=provider_config["temperature"],
                    num_predict=provider_config["max_tokens"],
                    timeout=provider_config["timeout"],
                )
                
                # Perform health check
                health_status, error = await self._check_model_health(
                    model_name, base_url
                )
                
                # Get capabilities if model is healthy
                # Note: langchain-ollama profile is often None, so we use Ollama API directly
                if health_status == "healthy":
                    try:
                        # Try to get capabilities from Ollama API (more reliable)
                        capabilities = await self._get_capabilities_from_api(
                            model_name, base_url
                        )
                        
                        # Also try to get from langchain model.profile if available
                        # (though it's usually None for Ollama models)
                        if langchain_model and hasattr(langchain_model, "profile"):
                            try:
                                profile = langchain_model.profile
                                if profile is not None:
                                    if isinstance(profile, dict):
                                        # Merge with API capabilities (profile takes precedence)
                                        capabilities = {**capabilities, **profile}
                                    elif hasattr(profile, "model_dump"):
                                        # Pydantic model
                                        profile_dict = profile.model_dump()
                                        capabilities = {**capabilities, **profile_dict}
                                    elif hasattr(profile, "__dict__"):
                                        # Object with __dict__
                                        capabilities = {**capabilities, **profile.__dict__}
                            except Exception as e:
                                logger.debug(f"Could not access langchain profile: {e}")
                        
                        # Ensure capabilities is a dict
                        if not isinstance(capabilities, dict):
                            capabilities = {}
                            
                    except Exception as e:
                        logger.warning(
                            f"Failed to get capabilities for {model_name}: {e}"
                        )
                        capabilities = {}
                
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {e}")
                health_status = "unhealthy"
                error = str(e)
            
            # Create ModelInstance
            model_instance = ModelInstance(
                model_name=model_name,
                provider_id=provider_id,
                config=provider_config,
                langchain_model=langchain_model,
                health_status=health_status,
                capabilities=capabilities,
                error=error,
            )
            
            # Store in dictionary
            self.models[provider_id] = model_instance
            
            logger.info(
                f"Initialized model {model_name} (provider_id={provider_id}): "
                f"health={health_status}"
            )
            
            return model_instance
            
        except Exception as e:
            logger.error(f"Error initializing model from {config_file}: {e}")
            return None
    
    async def _check_model_health(
        self, model_name: str, base_url: str, timeout: float = 5.0
    ) -> tuple[str, Optional[str]]:
        """Check if a model is healthy by checking if it's available in Ollama.
        
        Args:
            model_name: Model name to check
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (health_status, error_message)
            health_status: "healthy" or "unhealthy"
            error_message: None if healthy, error string if unhealthy
        """
        try:
            # Check if model is available by listing models (lighter than generating)
            async with httpx.AsyncClient(timeout=timeout) as client:
                # First, check if Ollama is reachable
                try:
                    health_response = await client.get(f"{base_url}/api/tags")
                    if health_response.status_code != 200:
                        return (
                            "unhealthy",
                            f"Ollama API returned status {health_response.status_code}",
                        )
                except httpx.ConnectError:
                    return ("unhealthy", "Connection error - Ollama not reachable")
                
                # Check if model exists in the list
                models_response = await client.get(f"{base_url}/api/tags")
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    available_models = [
                        model.get("name", "") for model in models_data.get("models", [])
                    ]
                    
                    # Check if our model is in the list
                    if model_name in available_models:
                        return ("healthy", None)
                    else:
                        return (
                            "unhealthy",
                            f"Model '{model_name}' not found in Ollama. Available: {', '.join(available_models[:5])}...",
                        )
                else:
                    return (
                        "unhealthy",
                        f"Failed to list models: status {models_response.status_code}",
                    )
                    
        except httpx.TimeoutException:
            return ("unhealthy", "Request timeout")
        except httpx.ConnectError:
            return ("unhealthy", "Connection error - Ollama not reachable")
        except Exception as e:
            return ("unhealthy", str(e))
    
    async def _get_capabilities_from_api(
        self, model_name: str, base_url: str
    ) -> Dict[str, Any]:
        """Get model capabilities from Ollama API.
        
        This extracts capabilities from Ollama's /api/show endpoint and
        infers capabilities based on model name and known features.
        
        Args:
            model_name: Model name
            base_url: Ollama API base URL
            
        Returns:
            Dictionary of capabilities including:
            - tool_calling: Whether model supports tool calling
            - image_inputs: Whether model supports image inputs
            - reasoning_output: Whether model supports reasoning
            - max_input_tokens: Context window size (if available)
        """
        capabilities = {}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get model info from Ollama API (POST request with JSON body)
                response = await client.post(
                    f"{base_url}/api/show",
                    json={"name": model_name}
                )
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract parameters (can be dict or string)
                    parameters = data.get("parameters", {})
                    num_ctx = None
                    
                    if isinstance(parameters, str):
                        # Parse string format: "key value\nkey2 value2"
                        params_dict = {}
                        for line in parameters.split("\n"):
                            line = line.strip()
                            if line:
                                # Handle format: "num_ctx                  32768"
                                # Split by whitespace, but handle multiple spaces
                                parts = line.split()
                                if len(parts) >= 2:
                                    key = parts[0]
                                    # Try to get numeric value (last part is usually the value)
                                    try:
                                        value = parts[-1]
                                        # Remove quotes if present
                                        value = value.strip('"\'')
                                        # Try to convert to number
                                        if "." in value:
                                            params_dict[key] = float(value)
                                        else:
                                            params_dict[key] = int(value)
                                    except (ValueError, IndexError):
                                        # If conversion fails, join remaining parts as string
                                        params_dict[key] = " ".join(parts[1:])
                        
                        # Extract num_ctx from parsed dict
                        num_ctx = params_dict.get("num_ctx")
                        parameters = params_dict
                    elif isinstance(parameters, dict):
                        num_ctx = parameters.get("num_ctx")
                    
                    # Get context window size from parameters
                    if num_ctx:
                        try:
                            capabilities["max_input_tokens"] = int(num_ctx)
                        except (ValueError, TypeError):
                            pass
                    
                    # Check if Ollama API provides capabilities directly
                    # Ollama returns capabilities as a list: ['completion', 'tools', 'thinking']
                    api_capabilities = data.get("capabilities", [])
                    if isinstance(api_capabilities, list):
                        if "tools" in api_capabilities:
                            capabilities["tool_calling"] = True
                        if "thinking" in api_capabilities:
                            capabilities["reasoning_output"] = True
                        if "completion" in api_capabilities:
                            capabilities["completion"] = True
                    elif isinstance(api_capabilities, dict):
                        capabilities.update(api_capabilities)
                    
                    # Check details for additional info
                    details = data.get("details", {})
                    if details:
                        # Extract family info which might indicate capabilities
                        family = details.get("family", "")
                        if family:
                            capabilities["family"] = family
                    
                    # Check modelfile for hints about capabilities
                    modelfile = data.get("modelfile", "")
                    
                    # Infer capabilities based on model name and known features
                    # Models known to support tool calling
                    tool_calling_models = [
                        "qwen3", "qwen2.5", "qwen2", "deepseek", "devstral",
                        "granite4", "command-r", "mistral-small", "mistral:latest", "llama4"
                    ]
                    model_lower = model_name.lower()
                    if any(tc_model in model_lower for tc_model in tool_calling_models):
                        capabilities["tool_calling"] = True
                    
                    # Models known to support reasoning
                    reasoning_models = ["deepseek-r1", "qwen2.5-reasoner"]
                    if any(rm in model_lower for rm in reasoning_models):
                        capabilities["reasoning_output"] = True
                    
                    # Models known to support images (multimodal)
                    image_models = ["llava", "bakllava", "moondream", "minicpm-v"]
                    if any(im in model_lower for im in image_models):
                        capabilities["image_inputs"] = True
                    
                    # Store additional metadata
                    capabilities["model_name"] = model_name
                    
        except Exception as e:
            logger.debug(f"Could not get capabilities from API for {model_name}: {e}")
        
        return capabilities
    
    def get_model(self, provider_id: str) -> Optional[ModelInstance]:
        """Get a pre-initialized model instance.
        
        Args:
            provider_id: Provider ID (e.g., "ollama.qwen")
            
        Returns:
            ModelInstance or None if not found
        """
        return self.models.get(provider_id)
    
    def get_all_models(self) -> Dict[str, ModelInstance]:
        """Get all initialized models.
        
        Returns:
            Dictionary of provider_id -> ModelInstance
        """
        return self.models.copy()
    
    def get_healthy_models(self) -> Dict[str, ModelInstance]:
        """Get only healthy models.
        
        Returns:
            Dictionary of provider_id -> ModelInstance (only healthy ones)
        """
        return {
            pid: model
            for pid, model in self.models.items()
            if model.health_status == "healthy"
        }


# Global model manager singleton
_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get global model manager instance.
    
    Returns:
        ModelManager singleton
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


async def initialize_model_manager(config_dir: Path = Path("config")) -> ModelManager:
    """Initialize global model manager with all models.
    
    Args:
        config_dir: Directory containing agent.*.yaml config files
        
    Returns:
        Initialized ModelManager
    """
    manager = get_model_manager()
    await manager.initialize_all_models(config_dir)
    return manager

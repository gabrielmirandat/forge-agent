"""Ollama LLM provider implementation.

This provider integrates with Ollama for local LLM inference.
Optimized for qwen2.5-coder:7b with low temperature for deterministic planning.
"""

import json
from typing import Any, Dict, List

import httpx

from agent.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local inference."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider.

        Args:
            config: Configuration dict with:
                - base_url: Ollama API URL (default: http://localhost:11434)
                - model: Model name (default: qwen2.5-coder:7b)
                - temperature: Temperature (default: 0.1)
                - timeout: Request timeout in seconds (default: 300)
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model = config.get("model", "qwen2.5-coder:7b")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)  # Store max_tokens from config
        self.timeout = config.get("timeout", 300)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text

        Raises:
            httpx.HTTPError: If request fails
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate response from chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters:
                - temperature: Override default temperature
                - max_tokens: Max tokens to generate
                - session_id: Optional session ID for metrics tracking

        Returns:
            Generated response text

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is invalid
        """
        # Use kwargs temperature if provided, otherwise use instance default
        temperature = kwargs.get("temperature", self.temperature)
        # Use max_tokens from kwargs if provided, otherwise use instance default from config
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        session_id = kwargs.get("session_id")

        # Build request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        # Make request
        import time
        request_start = time.time()
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                
                request_duration = time.time() - request_start
                import logging
                logging.getLogger("ollama.timing").info(
                    f"Ollama request completed in {request_duration:.2f}s "
                    f"(model={self.model}, messages={len(messages)})"
                )

                data = response.json()

                # Extract message content
                if "message" in data and "content" in data["message"]:
                    content = data["message"]["content"]
                    
                    # Track usage metrics if session_id provided
                    if session_id:
                        try:
                            from agent.observability.llm_metrics import get_llm_metrics
                            
                            # Extract token usage from Ollama response
                            # Ollama returns: prompt_eval_count (prompt tokens), eval_count (completion tokens)
                            prompt_tokens = data.get("prompt_eval_count", 0) or 0
                            completion_tokens = data.get("eval_count", 0) or 0
                            total_tokens = prompt_tokens + completion_tokens
                            
                            # Also check for usage object (some Ollama versions use this)
                            if "usage" in data:
                                usage = data["usage"]
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                completion_tokens = usage.get("completion_tokens", completion_tokens)
                                total_tokens = usage.get("total_tokens", total_tokens)
                            
                            # Debug: log what we're getting from Ollama
                            import logging
                            logger = logging.getLogger("ollama.metrics")
                            logger.debug(
                                f"Ollama response data keys: {list(data.keys())}, "
                                f"prompt_eval_count: {data.get('prompt_eval_count')}, "
                                f"eval_count: {data.get('eval_count')}, "
                                f"usage: {data.get('usage')}"
                            )
                            
                            # Record usage (even if tokens are 0, we still want to track the call)
                            metrics = get_llm_metrics()
                            metrics.record_usage(
                                session_id=session_id,
                                model=self.model,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens if total_tokens > 0 else None,
                            )
                            
                            logger.debug(
                                f"Recorded LLM usage for session {session_id}: "
                                f"model={self.model}, tokens={total_tokens}, calls=1"
                            )
                        except Exception as e:
                            # Log error but don't fail if metrics tracking fails
                            import logging
                            logging.getLogger("ollama.metrics").warning(f"Failed to track LLM metrics: {e}", exc_info=True)
                    
                    return content
                else:
                    raise ValueError(f"Unexpected response format: {data}")

            except httpx.HTTPStatusError as e:
                raise httpx.HTTPError(
                    f"Ollama API error: {e.response.status_code} - {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise httpx.HTTPError(f"Request failed: {e}") from e


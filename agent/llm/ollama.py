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

        Returns:
            Generated response text

        Raises:
            httpx.HTTPError: If request fails
            ValueError: If response is invalid
        """
        # Use kwargs temperature if provided, otherwise use instance default
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", 2048)

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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

                data = response.json()

                # Extract message content
                if "message" in data and "content" in data["message"]:
                    return data["message"]["content"]
                else:
                    raise ValueError(f"Unexpected response format: {data}")

            except httpx.HTTPStatusError as e:
                raise httpx.HTTPError(
                    f"Ollama API error: {e.response.status_code} - {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise httpx.HTTPError(f"Request failed: {e}") from e


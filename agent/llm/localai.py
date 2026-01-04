"""LocalAI LLM provider implementation."""

import httpx
from typing import Any, Dict, List

from agent.llm.base import LLMProvider


class LocalAIProvider(LLMProvider):
    """LocalAI provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize LocalAI provider.

        Args:
            config: Configuration with base_url, model, temperature, etc.
        """
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8080")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.timeout = config.get("timeout", 300)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Convert single prompt to chat format
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate response from chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        # TODO: Implement LocalAI API call
        # - Use httpx to call LocalAI API
        # - Handle errors and timeouts
        # - Return generated text
        raise NotImplementedError("LocalAI chat not yet implemented")


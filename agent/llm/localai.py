"""LocalAI LLM provider implementation.

This provider implements the LLM interface for LocalAI. The LLM is used
exclusively as a reasoning engine - it never executes code or performs
system actions. It only returns structured text outputs.
"""

import httpx
from typing import Any, Dict, List

from agent.llm.base import LLMProvider


class LocalAIProvider(LLMProvider):
    """LocalAI provider implementation.

    This is a reasoning engine only. It never executes code, accesses the
    filesystem, or performs system actions.
    """

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

        This method only returns text. It does NOT execute any actions.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text (structured output for planning)
        """
        # Convert single prompt to chat format
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate response from chat messages.

        This method only returns text. It does NOT execute any actions.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            Generated response (structured output for planning)
        """
        # TODO: Implement LocalAI API call
        # - Use httpx to call LocalAI API
        # - Handle errors and timeouts
        # - Return generated text only (no execution)
        raise NotImplementedError("LocalAI chat not yet implemented")


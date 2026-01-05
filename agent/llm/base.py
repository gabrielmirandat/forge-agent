"""Base LLM provider interface.

The LLM is treated as an untrusted but useful reasoning component. It:
- Never executes code
- Never accesses the filesystem directly
- Never performs network or system actions
- Only returns structured text outputs

The LLM is used exclusively by the Planner for reasoning and plan generation.
It has no direct connection to tools or execution.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMProvider(ABC):
    """Base class for LLM providers.

    The LLM is a reasoning engine only. It never executes code, accesses the
    filesystem, or performs system actions. It only returns structured text
    outputs that are consumed by the Planner.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM provider.

        Args:
            config: Provider-specific configuration
        """
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt.

        This method only returns text. It does NOT execute any actions.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text (structured output for planning)
        """
        pass

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Generate response from chat messages.

        This method only returns text. It does NOT execute any actions.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            Generated response (structured output for planning)
        """
        pass


"""Base LLM provider interface.

Supports both text generation and function calling (tool calling) like OpenCode.
The LLM can use tools directly via function calling, without requiring a planner.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM."""
    
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """Response from LLM that may contain text and/or tool calls."""
    
    content: Optional[str] = None
    tool_calls: List[ToolCall] = None
    
    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class LLMProvider(ABC):
    """Base class for LLM providers.

    Supports both text generation and function calling (tool calling).
    Like OpenCode, the LLM can use tools directly without requiring a planner.
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

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Union[str, LLMResponse]:
        """Generate response from chat messages with optional tool support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions for function calling
            **kwargs: Additional generation parameters

        Returns:
            Generated response (str for text-only, LLMResponse for tool calls)
        """
        pass


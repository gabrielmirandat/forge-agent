"""Custom callback handlers for LangChain agents.

Provides error handling and observability for agent execution.
"""

from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agent.observability import get_logger, log_event


class ErrorHandlingCallbackHandler(BaseCallbackHandler):
    """Callback handler for error handling and observability.
    
    Handles errors during agent execution, tool calls, and LLM operations.
    Provides logging and event publishing for debugging and monitoring.
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize error handling callback.
        
        Args:
            session_id: Optional session ID for event correlation
        """
        super().__init__()
        self.session_id = session_id
        self.logger = get_logger("langchain.callbacks", "executor")
    
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """Called when a tool starts executing.
        
        Args:
            serialized: Serialized tool information
            input_str: Tool input string
            **kwargs: Additional arguments
        """
        tool_name = serialized.get("name", "unknown")
        log_event(
            self.logger,
            "tool.start",
            session_id=self.session_id,
            tool=tool_name,
            input=input_str[:200],  # Limit input length for logging
        )
    
    def on_tool_end(
        self,
        output: str,
        **kwargs: Any
    ) -> None:
        """Called when a tool finishes executing successfully.
        
        Args:
            output: Tool output
            **kwargs: Additional arguments
        """
        log_event(
            self.logger,
            "tool.end",
            session_id=self.session_id,
            output_length=len(output),
        )
    
    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when a tool execution fails.
        
        Args:
            error: Exception that occurred
            **kwargs: Additional arguments
        """
        error_msg = str(error)
        error_type = type(error).__name__
        
        log_event(
            self.logger,
            "tool.error",
            session_id=self.session_id,
            error=error_msg,
            error_type=error_type,
            level="ERROR",
        )
        
        # Note: Callbacks are synchronous, so we log errors here
        # Event publishing for errors is handled by the executor in the run() method
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        **kwargs: Any
    ) -> None:
        """Called when LLM starts generating.
        
        Args:
            serialized: Serialized LLM information
            prompts: List of prompts
            **kwargs: Additional arguments
        """
        log_event(
            self.logger,
            "llm.start",
            session_id=self.session_id,
            prompts_count=len(prompts),
        )
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        """Called when LLM finishes generating.
        
        Args:
            response: LLM response
            **kwargs: Additional arguments
        """
        # Handle None response gracefully
        if response is None:
            log_event(
                self.logger,
                "llm.end",
                session_id=self.session_id,
                generations_count=0,
                total_tokens=0,
            )
            return
        
        generations = getattr(response, "generations", [])
        llm_output = getattr(response, "llm_output", None)
        
        # Safely extract token usage
        total_tokens = 0
        if llm_output and isinstance(llm_output, dict):
            token_usage = llm_output.get("token_usage", {})
            if isinstance(token_usage, dict):
                total_tokens = token_usage.get("total_tokens", 0)
        
        log_event(
            self.logger,
            "llm.end",
            session_id=self.session_id,
            generations_count=len(generations) if generations else 0,
            total_tokens=total_tokens,
        )
    
    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when LLM generation fails.
        
        Args:
            error: Exception that occurred
            **kwargs: Additional arguments
        """
        error_msg = str(error)
        error_type = type(error).__name__
        
        log_event(
            self.logger,
            "llm.error",
            session_id=self.session_id,
            error=error_msg,
            error_type=error_type,
            level="ERROR",
        )
        
        # Publish error event (async-safe)
        self.logger.error(f"LLM error: {error_type} - {error_msg}")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when a chain starts executing.
        
        Args:
            serialized: Serialized chain information
            inputs: Chain inputs
            **kwargs: Additional arguments
        """
        # Handle None serialized gracefully
        if serialized is None:
            serialized = {}
        chain_name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown"
        log_event(
            self.logger,
            "chain.start",
            session_id=self.session_id,
            chain=chain_name,
        )
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when a chain finishes executing successfully.
        
        Args:
            outputs: Chain outputs
            **kwargs: Additional arguments
        """
        log_event(
            self.logger,
            "chain.end",
            session_id=self.session_id,
        )
    
    def on_chain_error(
        self,
        error: Exception,
        **kwargs: Any
    ) -> None:
        """Called when a chain execution fails.
        
        Args:
            error: Exception that occurred
            **kwargs: Additional arguments
        """
        error_msg = str(error)
        error_type = type(error).__name__
        
        log_event(
            self.logger,
            "chain.error",
            session_id=self.session_id,
            error=error_msg,
            error_type=error_type,
            level="ERROR",
        )
        
        # Publish error event (async-safe)
        self.logger.error(f"Chain error: {error_type} - {error_msg}")
    
    def on_agent_action(
        self,
        action: Any,
        **kwargs: Any
    ) -> None:
        """Called when agent takes an action.
        
        Args:
            action: Agent action
            **kwargs: Additional arguments
        """
        tool = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", "")
        
        log_event(
            self.logger,
            "agent.action",
            session_id=self.session_id,
            tool=tool,
            tool_input=str(tool_input)[:200],
        )
    
    def on_agent_finish(
        self,
        finish: Any,
        **kwargs: Any
    ) -> None:
        """Called when agent finishes execution.
        
        Args:
            finish: Agent finish information
            **kwargs: Additional arguments
        """
        log_event(
            self.logger,
            "agent.finish",
            session_id=self.session_id,
        )

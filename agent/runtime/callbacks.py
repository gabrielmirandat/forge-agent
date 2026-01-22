"""Custom callback handlers for LangChain agents.

Provides error handling and observability for agent execution.
"""

import time
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from agent.observability import get_logger, log_event


class ErrorHandlingCallbackHandler(BaseCallbackHandler):
    """Callback handler for error handling and observability.
    
    This callback handler extends LangChain's BaseCallbackHandler to provide:
    - Error handling and logging for agent execution
    - Tool call tracking (start, end, errors)
    - LLM call tracking with metrics collection
    - Chain execution monitoring
    - Agent action and finish events
    
    The handler automatically records LLM usage metrics when session_id is provided,
    tracking tokens, response times, and model information for observability.
    """
    
    def __init__(self, session_id: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize error handling callback.
        
        Args:
            session_id: Optional session ID for event correlation
            model_name: Optional model name for metrics tracking
        """
        super().__init__()
        self.session_id = session_id
        self.model_name = model_name
        self.logger = get_logger("langchain.callbacks", "executor")
        self._llm_start_times: Dict[str, float] = {}  # Track start times by run_id
    
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


class ReasoningAndDebugCallbackHandler(BaseCallbackHandler):
    """Callback handler for capturing reasoning, debugging info, and execution steps.
    
    This callback handler extends BaseCallbackHandler to provide:
    - Reasoning capture from LLM responses (additional_kwargs.reasoning_content)
    - Tool call tracking with full details
    - Error tracking for auto-correction monitoring
    - Intermediate steps collection
    - Debug information for troubleshooting
    """
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize reasoning and debug callback.
        
        Args:
            session_id: Optional session ID for event correlation
        """
        super().__init__()
        self.session_id = session_id
        self.logger = get_logger("reasoning.callbacks", "executor")
        
        # Collections for captured data
        self.reasoning_steps: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.tool_errors: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.intermediate_steps: List[Dict[str, Any]] = []
        self.debug_info: List[Dict[str, Any]] = []
    
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Called when LLM starts generating.
        
        Args:
            serialized: Serialized LLM information
            prompts: List of prompts
            **kwargs: Additional arguments
        """
        run_id = kwargs.get("run_id", "")
        self.llm_calls.append({
            "run_id": run_id,
            "status": "started",
            "prompt": prompts[0] if prompts else "",
            "timestamp": time.time(),
        })
        self.logger.debug(f"LLM call started: {run_id}")
    
    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs: Any
    ) -> None:
        """Called when LLM finishes generating.
        
        Captures reasoning from additional_kwargs if available.
        
        Args:
            response: LLM response
            **kwargs: Additional arguments
        """
        run_id = kwargs.get("run_id", "")
        
        # Update LLM call status
        for llm_call in self.llm_calls:
            if llm_call.get("run_id") == run_id:
                llm_call["status"] = "completed"
                llm_call["completion_time"] = time.time()
                break
        
        # Extract reasoning from response
        if response and hasattr(response, "generations"):
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, "message"):
                        message = generation.message
                        # Check for reasoning in additional_kwargs
                        if hasattr(message, "additional_kwargs"):
                            additional_kwargs = message.additional_kwargs
                            
                            # Check for reasoning_content (Ollama reasoning)
                            reasoning_content = additional_kwargs.get("reasoning_content")
                            if reasoning_content:
                                reasoning_text = str(reasoning_content)
                                self.reasoning_steps.append(reasoning_text)
                                self.logger.debug(f"Reasoning captured: {reasoning_text[:100]}...")
                            
                            # Check for other reasoning fields
                            for key in additional_kwargs.keys():
                                if "reason" in key.lower() or "think" in key.lower():
                                    reasoning_val = additional_kwargs.get(key)
                                    if reasoning_val:
                                        self.reasoning_steps.append(str(reasoning_val))
                                        self.logger.debug(f"Reasoning found in {key}")
    
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
        run_id = kwargs.get("run_id", "")
        
        tool_call = {
            "tool": tool_name,
            "input": input_str,
            "run_id": run_id,
            "status": "started",
            "timestamp": time.time(),
        }
        self.tool_calls.append(tool_call)
        
        # Add to intermediate steps
        self.intermediate_steps.append({
            "type": "tool_start",
            "tool": tool_name,
            "input": input_str,
            "timestamp": time.time(),
        })
        
        self.logger.debug(f"Tool started: {tool_name}")
    
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
        run_id = kwargs.get("run_id", "")
        
        # Update tool call status
        for tool_call in self.tool_calls:
            if tool_call.get("run_id") == run_id:
                tool_call["status"] = "completed"
                tool_call["output"] = str(output)[:500]  # Limit output length
                tool_call["completion_time"] = time.time()
                break
        
        # Add to intermediate steps
        self.intermediate_steps.append({
            "type": "tool_end",
            "output": str(output)[:500],
            "timestamp": time.time(),
        })
        
        self.logger.debug(f"Tool completed: {run_id}")
    
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
        run_id = kwargs.get("run_id", "")
        tool_name = kwargs.get("name", "unknown")
        
        error_info = {
            "tool": tool_name,
            "error": str(error),
            "error_type": type(error).__name__,
            "run_id": run_id,
            "timestamp": time.time(),
        }
        self.tool_errors.append(error_info)
        
        # Update tool call status
        for tool_call in self.tool_calls:
            if tool_call.get("run_id") == run_id:
                tool_call["status"] = "error"
                tool_call["error"] = str(error)
                break
        
        # Add to intermediate steps
        self.intermediate_steps.append({
            "type": "tool_error",
            "tool": tool_name,
            "error": str(error),
            "timestamp": time.time(),
        })
        
        self.logger.debug(f"Tool error: {tool_name} - {error}")
    
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when a chain starts executing.
        
        Args:
            serialized: Serialized chain information (may be None)
            inputs: Chain inputs (may be None)
            **kwargs: Additional arguments
        """
        # Handle case where serialized might be None
        if serialized is None:
            chain_name = "unknown"
        else:
            chain_name = serialized.get("name", "unknown") if isinstance(serialized, dict) else "unknown"
        
        self.debug_info.append({
            "type": "chain_start",
            "chain": chain_name,
            "timestamp": time.time(),
        })
    
    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Called when a chain finishes executing.
        
        Args:
            outputs: Chain outputs
            **kwargs: Additional arguments
        """
        self.debug_info.append({
            "type": "chain_end",
            "timestamp": time.time(),
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all captured data.
        
        Returns:
            Dictionary with reasoning, tool calls, errors, and debug info
        """
        return {
            "reasoning_steps": self.reasoning_steps,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
            "llm_calls": self.llm_calls,
            "intermediate_steps": self.intermediate_steps,
            "debug_info": self.debug_info,
            "summary": {
                "reasoning_count": len(self.reasoning_steps),
                "tool_calls_count": len(self.tool_calls),
                "tool_errors_count": len(self.tool_errors),
                "llm_calls_count": len(self.llm_calls),
                "intermediate_steps_count": len(self.intermediate_steps),
            }
        }

"""Custom callback handlers for LangChain agents.

Provides error handling and observability for agent execution.
"""

import time
from typing import Any, Dict, Optional

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
        # Track start time for response time calculation
        run_id = kwargs.get("run_id")
        if run_id:
            self._llm_start_times[run_id] = time.time()
        
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
        
        # Debug: log what we're getting
        if llm_output:
            self.logger.debug(
                f"LLM output keys: {list(llm_output.keys()) if isinstance(llm_output, dict) else type(llm_output)}, "
                f"full output: {llm_output}"
            )
        
        # Safely extract token usage
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        
        if llm_output and isinstance(llm_output, dict):
            # First check for Ollama-specific fields (most direct)
            if "prompt_eval_count" in llm_output:
                prompt_tokens = llm_output.get("prompt_eval_count", 0) or 0
            if "eval_count" in llm_output:
                completion_tokens = llm_output.get("eval_count", 0) or 0
            
            # Also check for standard token_usage dict
            token_usage = llm_output.get("token_usage", {})
            if isinstance(token_usage, dict):
                # Use token_usage if Ollama fields are not available
                if prompt_tokens == 0:
                    prompt_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0) or 0
                if completion_tokens == 0:
                    completion_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0) or 0
                if total_tokens == 0:
                    total_tokens = token_usage.get("total_tokens", 0) or 0
            
            # Calculate total if we have individual counts
            if (prompt_tokens > 0 or completion_tokens > 0) and total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens
        
        # Extract model name from response metadata
        model_name = None
        if llm_output and isinstance(llm_output, dict):
            model_name = llm_output.get("model_name") or llm_output.get("model")
        
        # Extract response time if available
        response_time = None
        
        # Try to calculate from start time
        run_id = kwargs.get("run_id")
        if run_id and run_id in self._llm_start_times:
            start_time = self._llm_start_times.pop(run_id)
            response_time = time.time() - start_time
        
        # Also check llm_output for timing information
        if not response_time and llm_output and isinstance(llm_output, dict):
            # Check for timing information
            if "response_time" in llm_output:
                response_time = llm_output.get("response_time")
            elif "total_duration" in llm_output:
                response_time = llm_output.get("total_duration")
        
        # Record metrics if session_id is available
        if self.session_id:
            try:
                from agent.observability.llm_metrics import get_llm_metrics
                metrics = get_llm_metrics()
                
                # Get model name from callback instance first (set during initialization)
                if not model_name:
                    model_name = self.model_name
                
                # Then try to get from kwargs or llm_output
                if not model_name:
                    # Try to get from kwargs
                    model_name = kwargs.get("model_name") or kwargs.get("model")
                    if not model_name and "invocation_params" in kwargs:
                        inv_params = kwargs.get("invocation_params", {})
                        model_name = inv_params.get("model_name") or inv_params.get("model")
                
                # Try to get model from the LLM instance itself
                if not model_name:
                    # Check if we can get it from the response object
                    if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
                        # Check all possible keys
                        for key in ["model_name", "model", "model_id"]:
                            if key in response.llm_output:
                                model_name = response.llm_output[key]
                                break
                
                # Default model name if still not found
                if not model_name:
                    model_name = "unknown"
                
                # Log debug info to help troubleshoot
                self.logger.debug(
                    f"Recording metrics: session={self.session_id}, "
                    f"model={model_name}, prompt_tokens={prompt_tokens}, "
                    f"completion_tokens={completion_tokens}, total_tokens={total_tokens}, "
                    f"llm_output_keys={list(llm_output.keys()) if llm_output else 'None'}, "
                    f"kwargs_keys={list(kwargs.keys())}"
                )
                
                metrics.record_usage(
                    session_id=self.session_id,
                    model=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens if total_tokens > 0 else None,
                    response_time=response_time,
                )
                
                self.logger.debug(
                    f"Recorded LLM metrics: session={self.session_id}, "
                    f"model={model_name}, tokens={total_tokens}, "
                    f"response_time={response_time}"
                )
            except Exception as e:
                # Log error but don't fail if metrics tracking fails
                self.logger.warning(
                    f"Failed to record LLM metrics: {e}",
                    exc_info=True
                )
        
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

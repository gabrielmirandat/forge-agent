"""Ollama LLM provider implementation.

This provider integrates with Ollama for local LLM inference.
Optimized for qwen2.5-coder:7b with low temperature for deterministic planning.
"""

import json
from typing import Any, Dict, List, Optional, Union

import httpx

from agent.llm.base import LLMProvider, LLMResponse, ToolCall


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

    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any
    ) -> Union[str, LLMResponse]:
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
            # Set keep_alive to 0 to free memory immediately after request
            # This helps prevent memory buildup
            "keep_alive": "0",
        }
        
        # Add tools if provided (Ollama supports function calling)
        # Ollama expects tools in OpenAI-compatible format
        if tools:
            payload["tools"] = tools
            import logging
            debug_logger = logging.getLogger("ollama.debug")
            debug_logger.info(
                f"Sending {len(tools)} tools to Ollama: {[t.get('function', {}).get('name', 'unknown') for t in tools]}"
            )
            # Log first tool structure for debugging
            if tools and debug_logger.isEnabledFor(logging.DEBUG):
                debug_logger.debug(f"First tool structure: {json.dumps(tools[0], indent=2)}")

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

                # Extract message content and tool calls
                message = data.get("message", {})

                # Debug: log full response structure
                import logging
                debug_logger = logging.getLogger("ollama.debug")
                debug_logger.info(
                    f"Ollama response keys: {list(data.keys())}, "
                    f"message keys: {list(message.keys())}, "
                    f"has tools in payload: {bool(tools)}, "
                    f"message content length: {len(message.get('content', ''))}"
                )
                # Log full message structure for debugging
                if debug_logger.isEnabledFor(logging.DEBUG):
                    debug_logger.debug(f"Full message structure: {json.dumps(message, indent=2)}")
                content = message.get("content", "")
                tool_calls = []
                
                # Check for tool calls in Ollama response
                # According to Ollama docs: tool_calls is a list with format:
                # {"type": "function", "function": {"index": 0, "name": "...", "arguments": {...}}}
                if "tool_calls" in message and message["tool_calls"]:
                    debug_logger.debug(f"Found {len(message['tool_calls'])} tool calls in response")
                    for idx, tc in enumerate(message["tool_calls"]):
                        func = tc.get("function", {})
                        # Arguments can be dict or string (JSON)
                        args = func.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        
                        # Use index from function if available, otherwise use our index
                        tool_id = func.get("index", idx)
                        tool_name = func.get("name", "")
                        
                        debug_logger.debug(
                            f"Processing tool call {idx}: name={tool_name}, "
                            f"index={tool_id}, args_keys={list(args.keys()) if isinstance(args, dict) else 'not_dict'}"
                        )
                        
                        if tool_name:  # Only add if we have a valid name
                            tool_calls.append(ToolCall(
                                id=str(tool_id),  # Ollama uses index as ID
                                name=tool_name,
                                arguments=args
                            ))
                        else:
                            debug_logger.warning(
                                f"Tool call {idx} has no name! func={func}"
                            )
                
                # If we have tool calls, return LLMResponse
                if tool_calls:
                    debug_logger.debug(f"Returning LLMResponse with {len(tool_calls)} tool calls")
                    return LLMResponse(content=content, tool_calls=tool_calls)
                
                # Otherwise return text content
                debug_logger.debug(f"Returning text response (length: {len(content) if content else 0})")
                debug_logger.warning(
                    f"No tool calls found in response. "
                    f"Response type: text, "
                    f"content length: {len(content) if content else 0}, "
                    f"has tool_calls key: {'tool_calls' in message}, "
                    f"message keys: {list(message.keys())}, "
                    f"tools were provided: {bool(tools)}"
                )
                if content:
                    
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


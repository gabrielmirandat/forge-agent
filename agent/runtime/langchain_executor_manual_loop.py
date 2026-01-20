"""Alternative LangChain executor using manual tool calling loop.

This version uses direct LLM invocation with manual tool calling loop
instead of create_agent, since create_agent doesn't work correctly with Ollama.
"""

import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.config.loader import AgentConfig
from agent.observability import get_logger, log_event
from agent.runtime.bus import EventType, publish
from agent.tools.base import ToolRegistry


class ManualToolCallingExecutor:
    """Executor using manual tool calling loop with direct LLM invocation.
    
    This executor:
    1. Invokes LLM directly (which works with Ollama)
    2. Checks for tool calls in response
    3. Executes tools manually
    4. Feeds results back to LLM
    5. Repeats until no more tool calls
    """
    
    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        session_id: Optional[str] = None,
        max_iterations: int = 50,
    ):
        """Initialize manual tool calling executor.
        
        Args:
            config: Agent configuration
            tool_registry: Tool registry
            session_id: Optional session ID
            max_iterations: Maximum number of tool calling iterations
        """
        self.config = config
        self.tool_registry = tool_registry
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.logger = get_logger("manual_tool_executor", "executor")
        
        # Will be initialized lazily
        self.langchain_llm = None
        self.langchain_llm_with_tools = None
        self.langchain_tools = None
        self.system_prompt = None
    
    async def _ensure_initialized(self) -> None:
        """Lazily initialize LLM and tools."""
        if self.langchain_llm_with_tools is not None:
            return
        
        # Load tools
        self.langchain_tools = await self.tool_registry.get_langchain_tools(
            session_id=self.session_id,
            config=self.config,
        )
        
        # Create LLM (same as LangChainExecutor)
        from langchain_ollama import ChatOllama
        
        self.langchain_llm = ChatOllama(
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
            timeout=self.config.llm.timeout,
        )
        
        # Bind tools
        tool_choice = getattr(self.config.llm, "tool_choice", "auto")
        if tool_choice != "auto":
            try:
                self.langchain_llm_with_tools = self.langchain_llm.bind_tools(
                    self.langchain_tools,
                    tool_choice=tool_choice
                )
            except TypeError:
                self.langchain_llm_with_tools = self.langchain_llm.bind_tools(self.langchain_tools)
        else:
            self.langchain_llm_with_tools = self.langchain_llm.bind_tools(self.langchain_tools)
        
        # Build system prompt
        from agent.runtime.langchain_executor import LangChainExecutor
        self.system_prompt = await LangChainExecutor.format_system_prompt(
            self.config,
            self.langchain_tools
        )
        
        self.logger.info(
            f"✅ Initialized with {len(self.langchain_tools)} tools",
            extra={"session_id": self.session_id, "tools_count": len(self.langchain_tools)}
        )
    
    def _find_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Find a tool by name."""
        for tool in self.langchain_tools:
            if tool.name == tool_name:
                return tool
        return None
    
    async def run(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run manual tool calling loop.
        
        Args:
            user_message: User's message
            conversation_history: Optional conversation history
            
        Returns:
            Dictionary with response and execution history
        """
        start_time = time.time()
        
        await self._ensure_initialized()
        
        # Build messages
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        
        # Add current user message
        messages.append(HumanMessage(content=user_message))
        
        # Manual tool calling loop
        iteration = 0
        final_response = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Invoke LLM
            response = await self.langchain_llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                self.logger.info(
                    f"🔧 Tool calls found: {len(response.tool_calls)}",
                    extra={"session_id": self.session_id, "iteration": iteration}
                )
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", f"call_{iteration}")
                    
                    self.logger.info(
                        f"🔧 Executing tool: {tool_name}",
                        extra={"session_id": self.session_id, "tool": tool_name, "args": tool_args}
                    )
                    
                    await publish(EventType.TOOL_CALLED, {
                        "session_id": self.session_id,
                        "tool": tool_name,
                        "arguments": tool_args,
                    })
                    
                    # Find and execute tool
                    tool = self._find_tool_by_name(tool_name)
                    if tool:
                        try:
                            tool_result = await tool.ainvoke(tool_args)
                            
                            await publish(EventType.TOOL_RESULT, {
                                "session_id": self.session_id,
                                "output": str(tool_result)[:500],
                            })
                            
                            # Add tool result to messages
                            messages.append(ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_id
                            ))
                        except Exception as e:
                            error_msg = f"Tool execution error: {e}"
                            self.logger.error(
                                error_msg,
                                extra={"session_id": self.session_id, "tool": tool_name},
                                exc_info=True
                            )
                            messages.append(ToolMessage(
                                content=error_msg,
                                tool_call_id=tool_id
                            ))
                    else:
                        error_msg = f"Tool '{tool_name}' not found"
                        self.logger.warning(
                            error_msg,
                            extra={"session_id": self.session_id, "tool": tool_name}
                        )
                        messages.append(ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_id
                        ))
            else:
                # No tool calls - we're done
                final_response = response.content if hasattr(response, 'content') else str(response)
                break
        
        duration_ms = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "response": final_response or "No response generated",
            "execution_history": [],
            "duration_ms": duration_ms,
        }

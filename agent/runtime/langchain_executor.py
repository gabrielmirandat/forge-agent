"""LangChain-based executor using Tool-Calling agents.

Uses LangChain's create_agent (LangChain 1.2.6+).
Tool-calling agents:
- LLM directly calls tools via function calling (native model support)
- Agent graph manages the loop: model → tool calls → tool results → model
- Manual message history: keeps last N interactions for recent context
- More efficient than ReAct (no explicit Thought/Action/Observation loop)
- Better for production use
- Callbacks provide observability and error handling
"""

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.agents.output_parsers.tools import (
    ToolsAgentOutputParser,
    parse_ai_message_to_tool_action,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.outputs import ChatGeneration, Generation

from agent.config.loader import AgentConfig
from agent.observability import get_logger, log_event
from agent.runtime.bus import EventType, publish
from agent.runtime.callbacks import ErrorHandlingCallbackHandler, ReasoningAndDebugCallbackHandler
from agent.runtime.event_helpers import publish_if_session_exists
from agent.storage import Storage
from agent.tools.base import ToolRegistry
from agent.utils.json_parser import extract_json_from_text


class _JSONToolCallParserWrapper:
    """Wrapper for LLM that parses JSON tool calls from text responses.
    
    Some models (like hhao/qwen2.5-coder-tools) return tool calls as JSON text
    instead of structured tool_calls. This wrapper intercepts responses and
    converts JSON tool calls to structured format.
    
    This wrapper delegates all attributes to the wrapped LLM to maintain
    compatibility with LangChain's Runnable interface.
    """
    
    def __init__(self, llm: Any, logger: Any, session_id: Optional[str] = None):
        """Initialize JSON tool call parser wrapper.
        
        Args:
            llm: LangChain LLM instance (with tools bound)
            logger: Logger instance
            session_id: Optional session ID for logging
        """
        self._llm = llm
        self._logger = logger
        self._session_id = session_id
    
    def _inject_tool_calls(self, message: AIMessage) -> AIMessage:
        """Inject tool calls into AIMessage if JSON tool call is found in content.
        
        The ToolsAgentOutputParser checks:
        1. message.tool_calls (preferred)
        2. message.additional_kwargs.get("tool_calls") (fallback)
        
        We inject into both to ensure compatibility.
        
        Args:
            message: AIMessage that may contain JSON tool call in content
            
        Returns:
            AIMessage with tool_calls injected if JSON was found
        """
        if not hasattr(message, "content") or not message.content:
            return message
        
        content = str(message.content)
        
        # Check if content contains JSON tool call pattern
        json_tool_call = extract_json_from_text(content)
        
        if json_tool_call:
            tool_name = json_tool_call.get("name", "")
            tool_args = json_tool_call.get("arguments", {})
            
            # Check if tool_calls already exists and is empty
            if not (hasattr(message, "tool_calls") and message.tool_calls):
                # Inject tool call in LangChain format
                tool_call = {
                    "name": tool_name,
                    "args": tool_args,
                    "id": f"parsed_{int(time.time() * 1000)}",
                    "type": "tool_call"
                }
                
                # Also inject in OpenAI format for additional_kwargs (ToolsAgentOutputParser checks this)
                openai_tool_call = {
                    "id": tool_call["id"],
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args) if not isinstance(tool_args, str) else tool_args
                    }
                }
                
                # Get existing additional_kwargs or create new dict
                additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
                if "tool_calls" not in additional_kwargs:
                    additional_kwargs["tool_calls"] = []
                additional_kwargs["tool_calls"].append(openai_tool_call)
                
                # Create new AIMessage with tool_calls and additional_kwargs
                try:
                    new_message = message.model_copy(update={
                        "tool_calls": [tool_call],
                        "additional_kwargs": additional_kwargs
                    })
                except Exception:
                    # Fallback: create new AIMessage
                    new_message = AIMessage(
                        content=message.content,
                        tool_calls=[tool_call],
                        additional_kwargs=additional_kwargs,
                        response_metadata=getattr(message, "response_metadata", {})
                    )
                
                self._logger.info(
                    f"🔧 Parsed and injected JSON tool call: {tool_name}",
                    extra={
                        "session_id": self._session_id,
                        "tool": tool_name,
                        "tool_args": tool_args,  # Changed from "args" to avoid conflict with logging
                    }
                )
                
                return new_message
        
        return message
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped LLM."""
        return getattr(self._llm, name)
    
    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        """Invoke LLM and parse JSON tool calls from response.
        
        Args:
            input: Input to LLM
            config: Optional config
            **kwargs: Additional arguments
            
        Returns:
            LLM response with tool_calls injected if JSON was found
        """
        self._logger.debug(
            "🔍 Wrapper intercepting ainvoke call",
            extra={"session_id": self._session_id}
        )
        response = await self._llm.ainvoke(input, config=config, **kwargs)
        
        # If response is an AIMessage, try to inject tool calls
        if isinstance(response, AIMessage):
            self._logger.debug(
                f"🔍 Wrapper processing AIMessage response",
                extra={"session_id": self._session_id, "content_preview": str(response.content)[:100]}
            )
            return self._inject_tool_calls(response)
        
        return response
    
    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        """Invoke LLM synchronously and parse JSON tool calls from response.
        
        Args:
            input: Input to LLM
            config: Optional config
            **kwargs: Additional arguments
            
        Returns:
            LLM response with tool_calls injected if JSON was found
        """
        response = self._llm.invoke(input, config=config, **kwargs)
        
        # If response is an AIMessage, try to inject tool calls
        if isinstance(response, AIMessage):
            return self._inject_tool_calls(response)
        
        return response
    
    async def astream(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        """Stream LLM response and parse JSON tool calls from chunks.
        
        Args:
            input: Input to LLM
            config: Optional config
            **kwargs: Additional arguments
            
        Yields:
            LLM response chunks with tool_calls injected if JSON was found
        """
        accumulated_content = ""
        async for chunk in self._llm.astream(input, config=config, **kwargs):
            if isinstance(chunk, AIMessage):
                accumulated_content += str(chunk.content) if chunk.content else ""
                # Check if we have a complete JSON tool call
                json_tool_call = extract_json_from_text(accumulated_content)
                if json_tool_call and not (hasattr(chunk, "tool_calls") and chunk.tool_calls):
                    # Inject tool call into chunk
                    tool_name = json_tool_call.get("name", "")
                    tool_args = json_tool_call.get("arguments", {})
                    tool_call = {
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"parsed_{int(time.time() * 1000)}",
                        "type": "tool_call"
                    }
                    try:
                        chunk = chunk.model_copy(update={"tool_calls": [tool_call]})
                    except Exception:
                        pass
            yield chunk


class JSONToolCallParser(ToolsAgentOutputParser):
    """Custom parser that extracts tool calls from JSON in message content.
    
    Extends ToolsAgentOutputParser to also check message.content for JSON tool calls
    when tool_calls and additional_kwargs are empty. This handles models that return
    JSON as text instead of structured tool_calls.
    """
    
    def __init__(self, logger: Any, session_id: Optional[str] = None, **kwargs: Any):
        """Initialize JSON tool call parser.
        
        Args:
            logger: Logger instance
            session_id: Optional session ID for logging
            **kwargs: Additional arguments for parent class
        """
        super().__init__(**kwargs)
        # Store logger and session_id in a way that doesn't conflict with Pydantic
        object.__setattr__(self, "_logger", logger)
        object.__setattr__(self, "_session_id", session_id)
    
    @property
    def logger(self) -> Any:
        """Get logger instance."""
        return getattr(self, "_logger", None)
    
    @property
    def session_id(self) -> Optional[str]:
        """Get session ID."""
        return getattr(self, "_session_id", None)
    
    def parse_result(
        self,
        result: list[Generation],
        *,
        partial: bool = False,
    ) -> list[AgentAction] | AgentFinish:
        """Parse result, checking content for JSON tool calls if needed.
        
        Args:
            result: List of generations
            partial: Whether this is a partial result
            
        Returns:
            List of AgentAction or AgentFinish
        """
        if not isinstance(result[0], ChatGeneration):
            msg = "This output parser only works on ChatGeneration output"
            raise ValueError(msg)
        
        message = result[0].message
        
        self.logger.debug(
            f"🔍 JSONToolCallParser.parse_result called",
            extra={
                "session_id": self._session_id,
                "has_tool_calls": bool(hasattr(message, "tool_calls") and message.tool_calls),
                "has_additional_kwargs": bool(hasattr(message, "additional_kwargs") and message.additional_kwargs),
                "content_preview": str(message.content)[:200] if hasattr(message, "content") else None,
            }
        )
        
        # First try the standard parsing (checks tool_calls and additional_kwargs)
        try:
            parsed = parse_ai_message_to_tool_action(message)
            
            self.logger.debug(
                f"🔍 Standard parsing result: {type(parsed).__name__}",
                extra={"session_id": self._session_id}
            )
            
            # If we got AgentFinish but content might have tool calls, check it
            if isinstance(parsed, AgentFinish):
                # Check if content contains JSON tool call
                if hasattr(message, "content") and message.content:
                    content_str = str(message.content)
                    json_tool_call = extract_json_from_text(content_str)
                    
                    if json_tool_call:
                        # Inject tool call into message and re-parse
                        tool_name = json_tool_call.get("name", "")
                        tool_args = json_tool_call.get("arguments", {})
                        
                        self.logger.info(
                            f"🔧 Found JSON tool call in content: {tool_name}",
                            extra={
                                "session_id": self._session_id,
                                "tool": tool_name,
                                "tool_args": tool_args,  # Changed from "args" to avoid conflict with logging
                            }
                        )
                        
                        tool_call = {
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"parsed_{int(time.time() * 1000)}",
                            "type": "tool_call"
                        }
                        
                        # Also inject in OpenAI format for additional_kwargs
                        openai_tool_call = {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args) if not isinstance(tool_args, str) else tool_args
                            }
                        }
                        
                        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
                        if "tool_calls" not in additional_kwargs:
                            additional_kwargs["tool_calls"] = []
                        additional_kwargs["tool_calls"].append(openai_tool_call)
                        
                        # Create new message with injected tool calls
                        try:
                            new_message = message.model_copy(update={
                                "tool_calls": [tool_call],
                                "additional_kwargs": additional_kwargs
                            })
                        except Exception:
                            new_message = AIMessage(
                                content=message.content,
                                tool_calls=[tool_call],
                                additional_kwargs=additional_kwargs,
                                response_metadata=getattr(message, "response_metadata", {})
                            )
                        
                        self.logger.info(
                            f"✅ Injected tool call and re-parsing: {tool_name}",
                            extra={
                                "session_id": self._session_id,
                                "tool": tool_name,
                            }
                        )
                        
                        # Re-parse with injected tool calls
                        re_parsed = parse_ai_message_to_tool_action(new_message)
                        self.logger.debug(
                            f"🔍 Re-parsing result: {type(re_parsed).__name__}",
                            extra={"session_id": self._session_id}
                        )
                        return re_parsed
            
            return parsed
        except Exception as e:
            self.logger.warning(
                f"Error in JSON tool call parser: {e}",
                extra={"session_id": self._session_id},
                exc_info=True
            )
            # Fall back to standard parsing
            return parse_ai_message_to_tool_action(message)


class LangChainExecutor:
    """Executor using Tool-Calling agents with LangChain.
    
    Uses LangChain's create_agent (LangChain 1.2.6+) to create stateful agents
    that can invoke tools via function calling. The agent graph manages the
    execution loop: model → tool calls → tool results → model.
    
    This executor is more efficient than ReAct-style agents because:
    - No explicit Thought/Action/Observation loop required
    - Native tool calling support from the LLM
    - Better suited for production use with proper state management
    
    Features:
    - Automatic tool binding via bind_tools()
    - State management with AgentState and MemorySaver checkpointer
    - Callbacks for observability and error handling
    - Manual message history management for context
    """
    
    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        session_id: Optional[str] = None,  # Deprecated: use run(session_id=...) instead
        max_iterations: int = 50,
        buffer_window_size: int = 10,
        storage: Optional[Storage] = None,
    ) -> None:
        """Initialize Tool-Calling agent executor (shared instance).
        
        All LLM providers and tools are loaded/executed through LangChain.
        The executor uses create_agent to build a stateful agent graph that
        manages tool calling loops automatically.
        
        This executor is designed to be shared across multiple sessions.
        Pass session_id to run() method instead of __init__.
        
        Args:
            config: Agent configuration containing LLM settings and runtime options.
            tool_registry: Tool registry containing all available tools.
            session_id: Deprecated - pass session_id to run() instead. Kept for backward compatibility.
            max_iterations: Maximum number of agent iterations before stopping.
            buffer_window_size: Number of recent messages to keep in context buffer.
            storage: Optional Storage instance to check session existence before publishing events.
        """
        self.config = config
        self.tool_registry = tool_registry
        self._default_session_id = session_id  # Kept for backward compatibility
        self.max_iterations = max_iterations
        self.buffer_window_size = buffer_window_size
        self.storage = storage
        self.logger = get_logger("langchain_executor", "executor")
        
        # Configure LangSmith tracing if enabled
        self._configure_langsmith()
        
        # Only Ollama is supported - use ChatOllama directly from LangChain
        provider_name = config.llm.provider.lower()
        
        if provider_name != "ollama":
            raise ValueError(
                f"Unsupported LLM provider: {provider_name}. "
                "Only 'ollama' is supported. "
                "Configure provider: ollama in your config file."
            )
        
        # Use ChatOllama from langchain-ollama (preferred) or langchain-community (fallback)
        try:
            # Try langchain-ollama first (recommended, not deprecated)
            from langchain_ollama import ChatOllama
        except ImportError:
            # Fallback to langchain-community (deprecated but still works)
            try:
                from langchain_community.chat_models import ChatOllama
                self.logger.warning(
                    "Using deprecated ChatOllama from langchain-community. "
                    "Install langchain-ollama for better support: pip install langchain-ollama"
                )
            except ImportError:
                raise ImportError(
                    "Ollama integration requires either 'langchain-ollama' or 'langchain-community'. "
                    "Install with: pip install langchain-ollama"
                )
        
        # Try to use pre-initialized model from ModelManager
        # This allows fast model switching without re-initialization
        from agent.llm.model_manager import get_model_manager
        
        model_manager = get_model_manager()
        
        # Find the model instance that matches current config
        # We need to match by model name and base_url
        model_instance = None
        for provider_id, instance in model_manager.get_all_models().items():
            if (instance.model_name == config.llm.model and 
                instance.config.get("base_url") == (config.llm.base_url or "http://localhost:11434")):
                model_instance = instance
                break
        
        if model_instance and model_instance.langchain_model and model_instance.health_status == "healthy":
            # Use pre-initialized model
            self.langchain_llm = model_instance.langchain_model
            self.logger.info(
                f"✅ Using pre-initialized model: {config.llm.model} (from ModelManager)"
            )
        else:
            # Fallback: create new ChatOllama instance
            # Note: Docker Manager ensures Ollama container is running before this point
            # The base_url should point to the Docker Ollama container (default: http://localhost:11434)
            # Reference: https://docs.ollama.com/capabilities/tool-calling#python
            # Ollama supports tool calling natively - LangChain's bind_tools() will convert tools to Ollama format
            # IMPORTANT: Not all Ollama models support tool calling!
            # Models that support tools: qwen3:8b, devstral, granite4, command-r, etc.
            # Check: https://ollama.com/search?c=tools
            # Note: llama3.1 removed - tool calling does not work reliably
            # Create ChatOllama instance with optimal settings for tool calling
            # IMPORTANT: temperature=0.0 is recommended for deterministic JSON output in tool calls
            # Lower temperature = more predictable tool call formatting
            # num_gpu: Use GPU if available (set to -1 to use all available GPUs, or specific number)
            # Ollama will automatically detect and use GPU if available
            num_gpu = getattr(config.llm, 'num_gpu', -1)  # Default to -1 (use all GPUs) if not specified
            # num_ctx: context window size. Default Ollama context (4096) is too small when 60+ tool
            # schemas are included in the request. 8192 is the recommended minimum.
            num_ctx = getattr(config.llm, 'num_ctx', 8192)

            self.langchain_llm = ChatOllama(
                model=config.llm.model,
                base_url=config.llm.base_url or "http://localhost:11434",
                temperature=config.llm.temperature,  # Should be 0.0 for best tool calling results
                num_predict=config.llm.max_tokens,  # Ollama uses num_predict instead of max_tokens
                num_gpu=num_gpu,  # Use GPU if available (-1 = use all GPUs, 0 = CPU only)
                num_ctx=num_ctx,  # Context window; must fit system prompt + tool schemas + history
                timeout=config.llm.timeout,
            )
            self.logger.info(
                f"✅ Created new ChatOllama instance: {config.llm.model} (fallback, not in ModelManager)"
            )
        
        # Log connection info for debugging (only if we created a new instance)
        if not (model_instance and model_instance.langchain_model):
            num_gpu = getattr(config.llm, 'num_gpu', -1)
            self.logger.info(
                f"ChatOllama initialized: model={config.llm.model}, "
                f"base_url={config.llm.base_url or 'http://localhost:11434'}, "
                f"num_gpu={num_gpu}"
            )
        
        # Tools and agent executor are loaded lazily in the async context (run/_ensure_agent_initialized)
        # This avoids trying to patch or manage the event loop (e.g., uvloop) in __init__,
        # which caused errors when running inside FastAPI/uvicorn.
        self.langchain_tools: List[Any] = []
        self.langchain_llm_with_tools = self.langchain_llm
        self.prompt_template: Optional[ChatPromptTemplate] = None
        self.agent: Optional[Any] = None  # Agent created by create_tool_calling_agent
        self.agent_executor: Optional[Any] = None  # AgentExecutor for running the agent
        
        # Memory management: manual message history
        # LangChain 1.2.6 does not provide ConversationBufferWindowMemory/ConversationSummaryMemory
        # in the same way, so we manage message history manually.
        # Note: message_history is per-session, managed in run() method
        self.buffer_window_size = buffer_window_size
        
        # Callbacks are created dynamically per execution in run() method
        # This allows the executor to be shared across multiple sessions
        # Get LangSmith callbacks if tracing is enabled (these are stateless)
        # Initialize as None, will be loaded lazily on first use
        self.langsmith_callbacks: Optional[List[Any]] = None
    
    async def _safe_publish(
        self, 
        event_type: EventType, 
        properties: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> bool:
        """Publish event only if session still exists.
        
        This prevents publishing events for deleted sessions, avoiding
        unnecessary frontend requests and 404 errors.
        
        Args:
            event_type: Type of event to publish
            properties: Event properties (must include session_id)
            session_id: Optional session ID (if not in properties)
            
        Returns:
            True if event was published, False if session doesn't exist
        """
        # Ensure session_id is in properties
        if "session_id" not in properties and session_id:
            properties["session_id"] = session_id
        return await publish_if_session_exists(event_type, properties, self.storage)

    async def _ensure_agent_initialized(self, session_id: Optional[str] = None) -> None:
        """Lazily initialize the agent on first use (avoids event loop issues in __init__)."""
        if self.agent_executor is not None:
            return
        current_session_id = session_id or self._default_session_id
        await self._load_and_bind_tools(current_session_id)
        await self._build_prompt_template(current_session_id)
        await self._create_agent_executor(current_session_id)

    async def _load_and_bind_tools(self, session_id: str) -> None:
        """Load MCP tools from registry and bind them to the LLM."""
        self.langchain_tools = await self.tool_registry.get_langchain_tools(
            session_id=self._default_session_id,
            config=self.config,
        )
        self.logger.info(
            f"Initialized with {len(self.langchain_tools)} LangChain tools",
            extra={"tool_names": [tool.name for tool in self.langchain_tools[:10]]},
        )

        tool_choice = getattr(self.config.llm, "tool_choice", None) or "auto"
        try:
            if tool_choice != "auto":
                try:
                    self.langchain_llm_with_tools = self.langchain_llm.bind_tools(
                        self.langchain_tools, tool_choice=tool_choice
                    )
                    self.logger.info(
                        f"✅ Tools bound with tool_choice='{tool_choice}' - {len(self.langchain_tools)} tools",
                        extra={"session_id": session_id, "tool_choice": tool_choice},
                    )
                except TypeError:
                    self.logger.debug(
                        "Model doesn't support tool_choice, using default bind_tools()",
                        extra={"session_id": session_id},
                    )
                    self.langchain_llm_with_tools = self.langchain_llm.bind_tools(self.langchain_tools)
            else:
                self.langchain_llm_with_tools = self.langchain_llm.bind_tools(self.langchain_tools)

            self.logger.info(
                f"✅ Tools bound - {len(self.langchain_tools)} tools",
                extra={
                    "session_id": session_id,
                    "tools_count": len(self.langchain_tools),
                    "tool_names_sample": [t.name for t in self.langchain_tools[:20]],
                    "tool_choice": tool_choice,
                },
            )
            if hasattr(self.langchain_llm_with_tools, "bound_tools"):
                count = len(self.langchain_llm_with_tools.bound_tools or [])
                self.logger.info(f"✅ Verified: Model has {count} bound tools", extra={"session_id": session_id})
        except Exception as e:
            self.logger.warning(
                f"⚠️ bind_tools() failed: {e}, using model without bind_tools",
                extra={"session_id": session_id},
                exc_info=True,
            )
            self.langchain_llm_with_tools = self.langchain_llm

    async def _build_prompt_template(self, session_id: str) -> None:
        """Build the ChatPromptTemplate from the formatted system prompt."""
        system_prompt_str = await self._format_system_prompt()
        self.logger.info(
            "System prompt built",
            extra={"session_id": session_id, "prompt_length": len(system_prompt_str), "tools_count": len(self.langchain_tools)},
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_str),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

    async def _create_agent_executor(self, session_id: str) -> None:
        """Create the agent chain and AgentExecutor."""
        from langchain_core.runnables import RunnablePassthrough
        from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages

        try:
            custom_parser = JSONToolCallParser(self.logger, session_id)
            self.agent = (
                RunnablePassthrough.assign(
                    agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"]),
                )
                | self.prompt_template
                | self.langchain_llm_with_tools
                | custom_parser
            )
            self.logger.info("✅ Agent created with create_tool_calling_agent", extra={"session_id": session_id})
        except Exception as e:
            msg = f"Failed to create tool calling agent: {e}"
            self.logger.error(f"❌ {msg}", extra={"session_id": session_id}, exc_info=True)
            raise ValueError(msg) from e

        try:
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.langchain_tools,
                verbose=False,
                max_iterations=self.max_iterations,
                max_execution_time=300.0,
                early_stopping_method="force",
                handle_parsing_errors=True,
                handle_tool_errors=True,  # Feed ToolException back to LLM so it can retry
                return_intermediate_steps=True,
            )
            self.logger.info(
                f"✅ AgentExecutor created with {len(self.langchain_tools)} tools",
                extra={
                    "session_id": session_id,
                    "tools_count": len(self.langchain_tools),
                    "model_type": type(self.langchain_llm_with_tools).__name__,
                    "max_iterations": self.max_iterations,
                },
            )
        except Exception as e:
            msg = f"Failed to create AgentExecutor: {e}"
            self.logger.error(f"❌ {msg}", extra={"session_id": session_id}, exc_info=True)
            raise ValueError(msg) from e
    
    def _configure_langsmith(self):
        """Configure LangSmith tracing if enabled.
        
        LangSmith tracing is controlled via environment variables:
        - LANGSMITH_TRACING: Enable tracing (true/false)
        - LANGSMITH_API_KEY: LangSmith API key
        - LANGSMITH_PROJECT: Project name (optional)
        - LANGSMITH_WORKSPACE_ID: Workspace ID (optional)
        """
        # Check if LangSmith tracing is enabled
        langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        
        if langsmith_tracing:
            # Set LangChain environment variables for tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            
            # API key is required
            if not os.getenv("LANGSMITH_API_KEY"):
                self.logger.warning(
                    "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set. "
                    "LangSmith tracing will be disabled."
                )
                return
            
            # Optional: Set project name (defaults to "default" if not set)
            if os.getenv("LANGSMITH_PROJECT"):
                os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
            else:
                # Use default project name (session_id not available in __init__)
                project_name = "forge-agent"
                os.environ["LANGCHAIN_PROJECT"] = project_name
            
            # Optional: Set workspace ID
            if os.getenv("LANGSMITH_WORKSPACE_ID"):
                os.environ["LANGCHAIN_WORKSPACE_ID"] = os.getenv("LANGSMITH_WORKSPACE_ID")
            
            # Set API key
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
            
            self.logger.info(
                f"LangSmith tracing enabled for project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}"
            )
        else:
            # Explicitly disable if not enabled
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    def _get_langsmith_callbacks(self, session_id: Optional[str] = None) -> List[Any]:
        """Get LangSmith callbacks if tracing is enabled.
        
        Args:
            session_id: Optional session ID for trace organization.
        
        Returns:
            List of LangSmith callback handlers, or empty list if disabled
        """
        langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        
        if not langsmith_tracing or not os.getenv("LANGSMITH_API_KEY"):
            return []
        
        try:
            # Try langchain.callbacks.tracers first (LangChain v0.3+)
            try:
                from langchain.callbacks.tracers import LangChainTracer
            except ImportError:
                # Fallback to langsmith.tracers (older versions or direct import)
                try:
                    from langsmith import Client
                    from langchain.callbacks.tracers.langchain import LangChainTracer
                except ImportError:
                    # Last fallback: use environment variables only
                    # LangChain will automatically use LangSmith if env vars are set
                    self.logger.info(
                        "LangSmith tracing enabled via environment variables. "
                        "LangChain will automatically send traces to LangSmith."
                    )
                    return []
            
            # Create LangSmith tracer
            tracer = LangChainTracer()
            
            # Add metadata for better trace organization
            if session_id:
                # Set run name for better trace identification
                tracer.run_name = f"agent-execution-{session_id[:8]}"
            
            return [tracer]
        except ImportError as e:
            self.logger.warning(
                f"LangSmith tracing is enabled but required packages are not installed. "
                f"Install with: pip install langsmith. Error: {e}"
            )
            return []
        except Exception as e:
            # Log error but continue without LangSmith tracing
            # According to LangChain best practices, use msg variable for error messages
            msg = f"Failed to initialize LangSmith tracer: {str(e)}"
            self.logger.warning(msg)
            return []
    
    
    @staticmethod
    def _generate_server_summary(server_name: str, tools: List[str]) -> str:
        """Generate a detailed summary for an MCP server with usage guidance.
        
        Args:
            server_name: Name of the MCP server
            tools: List of tool names from this server
            
        Returns:
            Formatted summary string with when to use and available functions
        """
        server_lower = server_name.lower()
        
        # Detailed server summaries with usage guidance
        server_summaries = {
            "desktop_commander": {
                "when_to_use": "whenever you need filesystem and shell operations",
                "description": "read/write files, list directories, create folders, execute commands, manage processes",
            },
            "playwright": {
                "when_to_use": "whenever you need browser automation and e2e testing",
                "description": "navigate pages, click elements, fill forms, take screenshots, test web applications",
            },
            "git": {
                "when_to_use": "whenever you need git repository operations",
                "description": "status, commit, branch, log, diff, checkout, merge, and other version control tasks",
            },
            "github": {
                "when_to_use": "whenever you need GitHub API operations",
                "description": "manage repositories, issues, pull requests, and GitHub resources",
            },
            "openapi": {
                "when_to_use": "whenever you need OpenAPI/Swagger API operations",
                "description": "validate documents, generate code snippets, get API operations",
            },
            "python_refactoring": {
                "when_to_use": "whenever you need Python code refactoring and analysis",
                "description": "analyze code, find issues, refactor functions, improve code quality",
            },
            "fetch": {
                "when_to_use": "whenever you need to fetch web content or make HTTP requests",
                "description": "download files, fetch URLs, retrieve web content",
            },
            "filesystem": {
                "when_to_use": "whenever you need filesystem operations",
                "description": "list directories, read/write files, create/delete directories, move/rename files. IMPORTANT: Use /projects as the base path for all filesystem operations (e.g., /projects/forge-agent instead of ~/repos/forge-agent)",
            },
        }
        
        # Try to get detailed summary
        if server_name in server_summaries:
            summary = server_summaries[server_name]
        elif server_lower in server_summaries:
            summary = server_summaries[server_lower]
        else:
            # Generate generic summary
            if "git" in server_lower:
                summary = {
                    "when_to_use": "whenever you need git operations",
                    "description": "git repository operations and version control",
                }
            elif "browser" in server_lower or "playwright" in server_lower:
                summary = {
                    "when_to_use": "whenever you need browser automation",
                    "description": "browser automation and testing",
                }
            elif "file" in server_lower or "fs" in server_lower or "commander" in server_lower:
                summary = {
                    "when_to_use": "whenever you need filesystem operations",
                    "description": "filesystem and file operations",
                }
            elif "api" in server_lower or "openapi" in server_lower:
                summary = {
                    "when_to_use": "whenever you need API operations",
                    "description": "API operations and management",
                }
            elif "refactor" in server_lower or "python" in server_lower:
                summary = {
                    "when_to_use": "whenever you need code refactoring",
                    "description": "code refactoring and analysis",
                }
            elif "fetch" in server_lower or "web" in server_lower:
                summary = {
                    "when_to_use": "whenever you need web content fetching",
                    "description": "web content fetching and HTTP requests",
                }
            else:
                summary = {
                    "when_to_use": f"whenever you need {server_name.replace('_', ' ')} operations",
                    "description": f"{server_name.replace('_', ' ')} operations",
                }
        
        # Format: "server_name - whenever you need X - description - available functions: func1, func2, ..."
        tools_str = ", ".join(sorted(tools))
        return f"{server_name} - {summary['when_to_use']} - {summary['description']} - Available functions: {tools_str}"
    
    @staticmethod
    def _generate_server_description(server_name: str, tools: List[str]) -> str:
        """Generate a description for an MCP server based on its name and tools.
        
        Args:
            server_name: Name of the MCP server
            tools: List of tool names from this server
            
        Returns:
            Description string
        """
        # This method is kept for backward compatibility
        # Use _generate_server_summary for detailed summaries
        summary = LangChainExecutor._generate_server_summary(server_name, tools)
        # Extract just the description part (between second and third dash)
        parts = summary.split(" - ")
        if len(parts) >= 3:
            return parts[2]  # Return the description part
        return summary
    
    @staticmethod
    async def format_system_prompt(
        config: Any,
        langchain_tools: List[Any]
    ) -> str:
        """Format system prompt with actual loaded tool names.

        Template variables:
        - {workspace_base}: Workspace base path on the host (e.g. ~/repos)
        - {filesystem_root}: Container path used by the filesystem MCP tool (e.g. /projects)
        - {workspace_root}: Container path used by git/other MCP tools (e.g. /workspace)
        - {tools}: Actual MCP tool names grouped by server, e.g. "filesystem: filesystem_list_directory, ..."

        Args:
            config: AgentConfig instance
            langchain_tools: Loaded LangChain tools from MCP servers

        Returns:
            Formatted system prompt string
        """
        workspace_base = "~/repos"
        if hasattr(config, "workspace") and hasattr(config.workspace, "base_path"):
            workspace_base = config.workspace.base_path

        # Extract container-side mount paths from MCP config so the model knows
        # which path prefix to use when calling each tool.
        filesystem_root = "/projects"
        workspace_root = "/workspace"
        if hasattr(config, "mcp") and config.mcp:
            fs_cfg = config.mcp.get("filesystem", {})
            # args: ["/projects"] tells the MCP server which dir it can access
            if isinstance(fs_cfg.get("args"), list) and fs_cfg["args"]:
                filesystem_root = fs_cfg["args"][0]
            elif isinstance(fs_cfg.get("volumes"), list) and fs_cfg["volumes"]:
                # fallback: parse "host_path:/container_path" volume string
                vol = fs_cfg["volumes"][0]
                if ":" in vol:
                    filesystem_root = vol.split(":", 1)[1]

            git_cfg = config.mcp.get("git", {})
            if isinstance(git_cfg.get("volumes"), list) and git_cfg["volumes"]:
                vol = git_cfg["volumes"][0]
                if ":" in vol:
                    workspace_root = vol.split(":", 1)[1]

        # Build tool list from the actual loaded tools.
        # Some MCP servers prefix their tools (e.g. git_status, browser_close),
        # others use bare names (e.g. list_directory from the filesystem server).
        # Group by checking which MCP server name the tool name starts with;
        # fall back to the first word before "_" if no match.
        mcp_server_names: List[str] = sorted(
            config.mcp.keys() if hasattr(config, "mcp") and config.mcp else [],
            key=len, reverse=True  # longest first so "python_refactoring" matches before "python"
        )
        tools_by_server: Dict[str, List[str]] = {}
        for tool in langchain_tools:
            server = None
            for sname in mcp_server_names:
                if tool.name.startswith(sname + "_") or tool.name == sname:
                    server = sname
                    break
            if server is None:
                # Fall back: first word before "_"
                parts = tool.name.split("_", 1)
                server = parts[0] if len(parts) > 1 else "other"
            tools_by_server.setdefault(server, []).append(tool.name)

        tools_str = "\n".join(
            f"{server}: {', '.join(sorted(names))}"
            for server, names in sorted(tools_by_server.items())
        ) if tools_by_server else "No tools available"

        if hasattr(config, "system_prompt_template") and config.system_prompt_template:
            template = config.system_prompt_template
            # /no_think is a Qwen3 control token that disables the thinking chain.
            # Small models (1.7b, 0.6b) don't support it and echo it back to users.
            model_name = config.llm.model if hasattr(config, "llm") and config.llm else ""
            _small_model = any(s in model_name for s in ("1.7b", "0.6b", "1b", "2b"))
            if _small_model:
                template = template.replace("/no_think\n", "").replace("/no_think", "")
        else:
            template = (
                "You are an AI agent with access to MCP tools. "
                "Repositories are at {filesystem_root}/ (filesystem tool) "
                "and {workspace_root}/ (git/other tools).\n\n"
                "Available tools:\n{tools}\n\n"
                "Rules:\n"
                "- Always use tools for file/git operations. Never guess content.\n"
                "- Chain tools when needed: explore → read → modify → verify.\n"
                "- For conversational questions, respond directly without tools."
            )

        return template.format(
            workspace_base=workspace_base,
            filesystem_root=filesystem_root,
            workspace_root=workspace_root,
            tools=tools_str,
        )
    
    async def _format_system_prompt(self) -> str:
        """Format system prompt using instance tools and config.
        
        Returns:
            Formatted system prompt string
        """
        return await self.format_system_prompt(self.config, self.langchain_tools)
    
    async def _build_system_prompt(self) -> str:
        """Build the system prompt string for the agent.
        
        Creates a system prompt string that includes:
        - Instructions for the agent
        - Available tools information
        - Workspace context
        
        Returns:
            System prompt string
            
        Note: According to LangChain docs (https://docs.langchain.com/oss/python/langchain/models#example-nested-structures),
        when using bind_tools(), tool schemas are provided automatically via function calling.
        The LLM receives tool definitions automatically, so we focus the prompt on WHEN to use tools,
        not on listing all tool details.
        """
        # Format the prompt using the same logic
        system_prompt_str = await self._format_system_prompt()
        
        return system_prompt_str
    
    def _load_memory_variables(self) -> Dict[str, Any]:
        """Load memory variables from message history.
        
        Returns:
            Dictionary with chat_history (list of messages)
        """
        # Return recent messages (last N messages)
        recent_messages = self.message_history[-self.buffer_window_size:] if self.message_history else []
        
        return {
            "chat_history": recent_messages,
        }
    
    
    def _save_context(self, user_input: str, agent_output: str):
        """Save conversation context to message history.
        
        Args:
            user_input: User message
            agent_output: Agent response
        """
        # Add to message history
        self.message_history.append({
            "role": "user",
            "content": user_input,
        })
        self.message_history.append({
            "role": "assistant",
            "content": agent_output,
        })
        
        # Keep only last N messages (buffer window)
        if len(self.message_history) > self.buffer_window_size * 2:
            self.message_history = self.message_history[-self.buffer_window_size * 2:]
    
    async def run(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        storage: Optional[Storage] = None,
    ) -> Dict[str, Any]:
        """Run Tool-Calling agent execution.
        
        Uses agent graph to manage the loop:
        - Model requests tools when needed
        - Agent graph invokes tools
        - Results fed back to model
        - Loop continues until agent finishes all steps
        - Memory (buffer + summary) provides context
        
        Uses astream_events (LangChain 0.2+) for granular event tracking:
        - on_tool_start: Tool execution begins
        - on_tool_end: Tool execution completes with results
        - on_llm_new_token: Real-time token streaming for LLM responses
        - on_llm_end: LLM generation completes
        - on_chain_end: Agent execution completes
        
        Falls back to astream if astream_events is not available.
        
        Args:
            user_message: User's message/request
            conversation_history: Optional conversation history (for initializing memory)
            session_id: Session ID for this execution (required for event publishing)
            storage: Optional Storage instance (overrides instance storage if provided)
            
        Returns:
            Dictionary with final response and execution history
        """
        # Use provided session_id or fall back to default (backward compatibility)
        current_session_id = session_id or self._default_session_id
        current_storage = storage or self.storage
        
        # Create callbacks dynamically for this session
        model_name = self.config.llm.model if self.config.llm else None
        callback_handler = ErrorHandlingCallbackHandler(
            session_id=current_session_id,
            model_name=model_name
        )
        reasoning_callback = ReasoningAndDebugCallbackHandler(session_id=current_session_id)
        
        # Combine callbacks: error handling + reasoning/debug + LangSmith tracing
        all_callbacks = [callback_handler, reasoning_callback]
        # Load LangSmith callbacks lazily if not already loaded
        if not hasattr(self, 'langsmith_callbacks') or self.langsmith_callbacks is None:
            self.langsmith_callbacks = self._get_langsmith_callbacks(session_id=current_session_id)
        if self.langsmith_callbacks:
            all_callbacks.extend(self.langsmith_callbacks)
        
        # Per-session message history (reset for each execution)
        message_history: List[Dict[str, Any]] = []
        
        start_time = time.time()
        log_event(
            self.logger,
            "tool_calling_executor.started",
            session_id=current_session_id,
            max_iterations=self.max_iterations,
        )

        # Ensure tools and agent graph are initialized in this async context
        await self._ensure_agent_initialized(session_id=current_session_id)
        try:
            # Initialize memory from conversation history if provided
            if conversation_history:
                # Process messages and add to history
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"]:
                        message_history.append({
                            "role": role,
                            "content": content,
                        })
            
            # Load memory variables (recent messages) from per-session history
            chat_history = message_history[-self.buffer_window_size:] if message_history else []
            
            # Convert chat history to LangChain messages for AgentExecutor
            # AgentExecutor expects chat_history as a list of BaseMessage objects
            langchain_chat_history = []
            for msg_dict in chat_history:
                role = msg_dict.get("role", "")
                content = msg_dict.get("content", "")
                if role == "user":
                    langchain_chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_chat_history.append(AIMessage(content=content))
            
            # Build input for AgentExecutor
            # AgentExecutor expects a dict with "input" and optionally "chat_history"
            # Reference: _helpers/langchain/libs/langchain/langchain_classic/agents/tool_calling_agent/base.py

            # Pre-process: inject tool hints from config/hints.yaml
            # Models pick wrong tools due to semantic bias when 60+ tools are in context;
            # injecting a hint directly into the user message overrides that bias reliably.
            from agent.utils.hints_loader import get_hints_loader
            _filesystem_root = "/projects"
            _workspace_root = "/workspace"
            if hasattr(self.config, "mcp") and self.config.mcp:
                _fs_cfg = self.config.mcp.get("filesystem", {})
                if isinstance(_fs_cfg.get("args"), list) and _fs_cfg["args"]:
                    _filesystem_root = _fs_cfg["args"][0]
                _git_cfg = self.config.mcp.get("git", {})
                if isinstance(_git_cfg.get("volumes"), list) and _git_cfg["volumes"]:
                    _vol = _git_cfg["volumes"][0]
                    if ":" in _vol:
                        _workspace_root = _vol.split(":", 1)[1]
            _hints_loader = get_hints_loader()
            effective_message = _hints_loader.apply(
                user_message, _filesystem_root, _workspace_root
            )
            if effective_message != user_message:
                self.logger.debug(
                    "Hint injected into user message",
                    extra={"session_id": current_session_id, "hint_applied": True},
                )

            input_data: Dict[str, Any] = {
                "input": effective_message,
            }
            
            # Add chat history if available
            if langchain_chat_history:
                input_data["chat_history"] = langchain_chat_history
            
            log_event(
                self.logger,
                "tool_calling_executor.agent.invoking",
                session_id=current_session_id,
                buffer_messages=len(chat_history),
            )
            
            # Log available tools for debugging
            self.logger.info(
                f"Available tools for agent: {[t.name for t in self.langchain_tools[:10]]}",
                extra={"session_id": current_session_id, "total_tools": len(self.langchain_tools)}
            )
            self.logger.info(
                f"User message: {user_message[:200]}",
                extra={"session_id": current_session_id}
            )
            
            # Log system prompt being used
            if self.prompt_template:
                system_prompt_for_log = ""
                try:
                    # Extract system message from prompt template
                    for msg_template in self.prompt_template.messages:
                        if hasattr(msg_template, "prompt") and hasattr(msg_template.prompt, "template"):
                            system_prompt_for_log = msg_template.prompt.template
                            break
                except Exception:
                    pass
                
                system_prompt_debug = system_prompt_for_log[:500] if system_prompt_for_log else "No system prompt"
                self.logger.info(
                    f"System prompt (first 500 chars): {system_prompt_debug}",
                    extra={"session_id": current_session_id, "prompt_length": len(system_prompt_for_log)}
                )
            
            # Use AgentExecutor to run the agent with real-time streaming
            # AgentExecutor manages the tool calling loop automatically
            # Reference: _helpers/langchain/libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
            # We use astream_events for real-time streaming similar to LangChain issue #34654
            # Reference: https://github.com/langchain-ai/langchain/issues/34654
            final_output = None
            accumulated_content = ""
            intermediate_steps = []
            last_chain_output = None
            
            # Prepare config for agent invocation (callbacks for observability)
            # Add run_name to identify the chain in LangChain traces (prevents "None chain" message)
            chain_name = f"forge-agent-{current_session_id[:8]}" if current_session_id else "forge-agent"
            agent_config: Dict[str, Any] = {
                "callbacks": all_callbacks,
                "run_name": chain_name,  # Set chain name to avoid "None chain" in logs
                "tags": ["forge-agent", "tool-calling-agent"],
            }
            
            try:
                # Use astream_events for real-time streaming (similar to LangChain docs)
                # This allows us to stream reasoning, tool calls, and LLM tokens in real-time
                # astream_events yields events as they happen, allowing real-time updates
                async for event in self.agent_executor.astream_events(
                    input_data,
                    version="v2",
                    config=agent_config,
                ):
                    event_name = event.get("event", "")
                    event_data = event.get("data", {})
                    event_name_full = event.get("name", "")
                    
                    # Stream LLM tokens in real-time with content_blocks pattern
                    # Uses LangChain's native content_blocks (LangChain 1.0+)
                    # Each chunk has content_blocks with types: "reasoning", "tool_call_chunk", "text"
                    # Reference: https://docs.langchain.com/docs/use_cases/streaming
                    if event_name == "on_chat_model_stream":
                        chunk = event_data.get("chunk")
                        if chunk:
                            content_blocks = []
                            try:
                                # Try to use native content_blocks if available
                                if hasattr(chunk, "content_blocks") and chunk.content_blocks is not None:
                                    native_blocks = chunk.content_blocks
                                    
                                    # Convert to dict format for JSON serialization
                                    for block in native_blocks:
                                        if isinstance(block, dict):
                                            content_blocks.append(block)
                                        else:
                                            # Convert ContentBlock object to dict
                                            block_type = getattr(block, "type", "text")
                                            block_dict = {"type": block_type}
                                            if hasattr(block, "text"):
                                                block_dict["text"] = block.text
                                            if hasattr(block, "reasoning"):
                                                block_dict["reasoning"] = block.reasoning
                                            if hasattr(block, "tool_call"):
                                                block_dict["tool_call"] = block.tool_call
                                            if hasattr(block, "tool_call_chunk"):
                                                block_dict["tool_call_chunk"] = block.tool_call_chunk
                                            content_blocks.append(block_dict)
                                else:
                                    # Fallback: Extract text content directly from chunk
                                    if hasattr(chunk, "content") and chunk.content:
                                        text_content = str(chunk.content) if chunk.content else ""
                                        if text_content:
                                            content_blocks.append({
                                                "type": "text",
                                                "text": text_content
                                            })
                                
                                self.logger.debug(
                                    f"Streaming chunk: {len(content_blocks)} blocks, "
                                    f"types={[b.get('type') for b in content_blocks]}"
                                )
                                
                                # Process each content block and publish events
                                # Accumulate text tokens first
                                text_tokens = []
                                for block in content_blocks:
                                    if block["type"] == "reasoning" and (reasoning := block.get("reasoning")):
                                        # Publish reasoning block (only if session exists)
                                        await self._safe_publish(EventType.LLM_REASONING, {
                                            "session_id": current_session_id,
                                            "content": reasoning,
                                        }, current_session_id)
                                    elif block["type"] == "tool_call_chunk":
                                        # tool_call_chunk is a partial streaming event — do NOT publish TOOL_CALLED
                                        # here since the tool name/args may be incomplete.
                                        # The canonical TOOL_CALLED event is published in on_tool_start instead.
                                        pass
                                    elif block["type"] == "text":
                                        # Collect text tokens (don't publish individually to avoid duplication)
                                        token = block.get("text", "")
                                        if token:
                                            text_tokens.append(token)
                                            accumulated_content += token
                                
                                # Publish content_blocks event for structured streaming (includes all blocks)
                                # This is the single source of truth - frontend processes this
                                # Only publish if session still exists
                                if content_blocks:
                                    await self._safe_publish(EventType.LLM_STREAM_TOKEN, {
                                        "session_id": current_session_id,
                                        "content_blocks": content_blocks,  # Structured blocks pattern
                                        "accumulated": accumulated_content,
                                        # Also include token for backward compatibility (only text tokens)
                                        "token": "".join(text_tokens) if text_tokens else None,
                                    }, current_session_id)
                            except Exception as e:
                                self.logger.warning(
                                    f"Error processing content_blocks: {e}",
                                    exc_info=True
                                )
                                continue
                    
                    # Stream LLM start
                    elif event_name == "on_chat_model_start":
                        # Only publish if session still exists
                        await self._safe_publish(EventType.LLM_STREAM_START, {
                            "session_id": current_session_id,
                            "model": event_data.get("name", ""),
                        }, current_session_id)
                    
                    # Stream LLM end and capture reasoning
                    elif event_name == "on_chat_model_end":
                        llm_output = event_data.get("output", "")
                        reasoning_content = None
                        
                        # Extract reasoning if available
                        if hasattr(llm_output, "additional_kwargs"):
                            reasoning_content = llm_output.additional_kwargs.get("reasoning_content")
                        elif isinstance(llm_output, dict):
                            additional_kwargs = llm_output.get("additional_kwargs", {})
                            reasoning_content = additional_kwargs.get("reasoning_content")
                        
                        # Record LLM usage metrics
                        if current_session_id:
                            try:
                                from agent.observability.llm_metrics import get_llm_metrics
                                
                                # Extract token usage from response
                                prompt_tokens = 0
                                completion_tokens = 0
                                total_tokens = 0
                                
                                # Try multiple ways to get token usage from LangChain/Ollama response
                                # Method 1: From event_data response_metadata
                                response_metadata = event_data.get("response_metadata", {})
                                eval_duration_ns: Optional[int] = None
                                if response_metadata:
                                    # Ollama format (prompt_eval_count, eval_count, eval_duration)
                                    if "prompt_eval_count" in response_metadata:
                                        prompt_tokens = response_metadata.get("prompt_eval_count", 0) or 0
                                    if "eval_count" in response_metadata:
                                        completion_tokens = response_metadata.get("eval_count", 0) or 0
                                    # eval_duration is nanoseconds of pure generation time (no prompt eval)
                                    eval_duration_ns = response_metadata.get("eval_duration")
                                    total_tokens = prompt_tokens + completion_tokens

                                    # Also check for usage object
                                    if "usage" in response_metadata:
                                        usage = response_metadata["usage"]
                                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                                        total_tokens = usage.get("total_tokens", total_tokens)

                                # Method 2: From llm_output.response_metadata (AIMessage)
                                if hasattr(llm_output, "response_metadata"):
                                    metadata = llm_output.response_metadata
                                    if metadata:
                                        # Ollama format
                                        if "prompt_eval_count" in metadata and not prompt_tokens:
                                            prompt_tokens = metadata.get("prompt_eval_count", 0) or 0
                                        if "eval_count" in metadata and not completion_tokens:
                                            completion_tokens = metadata.get("eval_count", 0) or 0
                                        if prompt_tokens or completion_tokens:
                                            total_tokens = prompt_tokens + completion_tokens
                                        if eval_duration_ns is None:
                                            eval_duration_ns = metadata.get("eval_duration")

                                # Method 3: From llm_output.usage_metadata (if available)
                                if hasattr(llm_output, "usage_metadata"):
                                    usage_meta = llm_output.usage_metadata
                                    if usage_meta:
                                        prompt_tokens = getattr(usage_meta, "input_tokens", prompt_tokens) or prompt_tokens
                                        completion_tokens = getattr(usage_meta, "output_tokens", completion_tokens) or completion_tokens
                                        total_tokens = prompt_tokens + completion_tokens

                                # Calculate tokens/sec from Ollama eval_duration (nanoseconds)
                                tokens_per_second: Optional[float] = None
                                if eval_duration_ns and eval_duration_ns > 0 and completion_tokens > 0:
                                    tokens_per_second = round(completion_tokens / (eval_duration_ns / 1e9), 1)

                                # Get model name from event or config
                                model_name = event_data.get("name", "")
                                if not model_name and hasattr(llm_output, "response_metadata"):
                                    metadata = llm_output.response_metadata
                                    if metadata:
                                        model_name = metadata.get("model", "")
                                if not model_name:
                                    model_name = self.config.llm.model if self.config.llm else "unknown"

                                # Calculate response time if available
                                response_time = None
                                if hasattr(llm_output, "response_metadata"):
                                    metadata = llm_output.response_metadata
                                    if metadata and "response_time" in metadata:
                                        response_time = metadata["response_time"]

                                # Always record usage (even if tokens are 0) to track calls
                                # This ensures we at least count the number of calls
                                metrics = get_llm_metrics()
                                metrics.record_usage(
                                    session_id=current_session_id,
                                    model=model_name,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=total_tokens if total_tokens > 0 else None,
                                    response_time=response_time,
                                    tokens_per_second=tokens_per_second,
                                )
                                self.logger.debug(
                                    f"Recorded LLM metrics: model={model_name}, "
                                    f"tokens={total_tokens}, tps={tokens_per_second}, session={current_session_id}"
                                )
                            except Exception as e:
                                self.logger.warning(f"Failed to track LLM metrics: {e}", exc_info=True)
                        
                        # Only publish if session still exists
                        await self._safe_publish(EventType.LLM_STREAM_END, {
                            "session_id": current_session_id,
                            "reasoning": str(reasoning_content) if reasoning_content else None,
                        }, current_session_id)
                        
                        # Also publish reasoning separately if found (only if session exists)
                        if reasoning_content:
                            await self._safe_publish(EventType.LLM_REASONING, {
                                "session_id": current_session_id,
                                "content": str(reasoning_content),
                            }, current_session_id)
                    
                    # Stream tool start
                    elif event_name == "on_tool_start":
                        tool_name = event.get("name", "") or event_name_full or event_data.get("name", "")
                        tool_input = event_data.get("input", {})
                        
                        # Only publish if session still exists
                        await self._safe_publish(EventType.TOOL_STREAM_START, {
                            "session_id": current_session_id,
                            "tool": tool_name,
                            "input": tool_input,
                        }, current_session_id)
                        
                        # Also publish as regular tool called event (only if session exists)
                        await self._safe_publish(EventType.TOOL_CALLED, {
                            "session_id": current_session_id,
                            "tool": tool_name,
                            "arguments": tool_input,
                        }, current_session_id)
                    
                    # Stream tool end
                    elif event_name == "on_tool_end":
                        tool_name = event_data.get("name", "") or event.get("name", "")
                        tool_output = event_data.get("output", "")
                        
                        # Only publish if session still exists
                        await self._safe_publish(EventType.TOOL_STREAM_END, {
                            "session_id": current_session_id,
                            "tool": tool_name,
                            "output": str(tool_output)[:500],  # Limit output length
                        }, current_session_id)
                        
                        # Also publish tool result (only if session exists)
                        await self._safe_publish(EventType.TOOL_RESULT, {
                            "session_id": current_session_id,
                            "tool": tool_name,
                            "output": str(tool_output)[:500],
                            "success": True,
                        }, current_session_id)
                    
                    # Stream tool error
                    elif event_name == "on_tool_error":
                        tool_name = event_data.get("name", "") or event.get("name", "")
                        error = event_data.get("error", "")
                        
                        # Only publish if session still exists
                        await self._safe_publish(EventType.TOOL_STREAM_ERROR, {
                            "session_id": current_session_id,
                            "tool": tool_name,
                            "error": str(error),
                        }, current_session_id)
                        
                        # Also publish as tool result with error (only if session exists)
                        await self._safe_publish(EventType.TOOL_RESULT, {
                            "session_id": current_session_id,
                            "tool": tool_name,
                            "error": str(error),
                            "success": False,
                        }, current_session_id)
                    
                    # Stream chain events - capture final output from AgentExecutor chain
                    elif event_name == "on_chain_start":
                        chain_name = event_name_full or event_data.get("name", "")
                        if "AgentExecutor" in chain_name:
                            # Only publish if session still exists
                            await self._safe_publish(EventType.EXECUTION_STARTED, {
                                "session_id": current_session_id,
                            }, current_session_id)
                        # Only publish if session still exists
                        await self._safe_publish(EventType.CHAIN_STREAM_START, {
                            "session_id": current_session_id,
                            "chain": chain_name,
                        }, current_session_id)
                    
                    elif event_name == "on_chain_end":
                        chain_name = event_name_full or event_data.get("name", "")
                        chain_output = event_data.get("output", {})
                        
                        # Capture final output from AgentExecutor
                        if "AgentExecutor" in chain_name:
                            if isinstance(chain_output, dict):
                                if "output" in chain_output:
                                    final_output = chain_output["output"]
                                    last_chain_output = chain_output
                                elif "messages" in chain_output:
                                    # Extract from messages
                                    messages = chain_output["messages"]
                                    for msg in reversed(messages):
                                        if hasattr(msg, "content") and msg.content:
                                            final_output = msg.content
                                            break
                                        elif isinstance(msg, dict) and "content" in msg:
                                            final_output = msg["content"]
                                            break
                            
                            # Only publish if session still exists
                            await self._safe_publish(EventType.EXECUTION_COMPLETED, {
                                "session_id": current_session_id,
                                "success": True,
                            }, current_session_id)
                        
                        # Only publish if session still exists
                        await self._safe_publish(EventType.CHAIN_STREAM_END, {
                            "session_id": current_session_id,
                            "chain": chain_name,
                        }, current_session_id)
                
                # Use last_chain_output as result if available, otherwise use accumulated_content
                if last_chain_output:
                    result = last_chain_output
                elif accumulated_content:
                    # Use accumulated_content directly as final_output, don't wrap in dict
                    final_output = accumulated_content
                    result = {"output": accumulated_content}  # Keep for intermediate_steps extraction
                else:
                    # Fallback: invoke to get result (shouldn't happen with astream_events)
                    self.logger.warning("No output from astream_events, falling back to invoke")
                    result = await asyncio.to_thread(
                        self.agent_executor.invoke,
                        input_data,
                        config=agent_config,
                    )
                
                self.logger.debug(
                    f"AgentExecutor result type={type(result)}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}",
                    extra={"session_id": current_session_id}
                )
                
                # Extract final output and intermediate steps from AgentExecutor result
                # AgentExecutor returns a dict with "output" key containing the final response
                # and "intermediate_steps" if return_intermediate_steps=True
                # Reference: _helpers/langchain/libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
                intermediate_steps = []
                if isinstance(result, dict):
                    # AgentExecutor typically returns {"output": "...", "intermediate_steps": [...]}
                    if "output" in result:
                        output_value = result["output"]
                        # Ensure we extract text, not a dict
                        if isinstance(output_value, str):
                            final_output = output_value
                        elif isinstance(output_value, dict):
                            # If output is a dict, try to extract text from it
                            final_output = output_value.get("content") or output_value.get("text") or str(output_value)
                            self.logger.warning(
                                f"Output is a dict, extracted text: {type(output_value)}",
                                extra={"session_id": current_session_id}
                            )
                        else:
                            final_output = str(output_value)
                        
                        self.logger.info(
                            f"✅ Extracted final output from AgentExecutor result['output']: {len(final_output) if final_output else 0} chars",
                            extra={"session_id": current_session_id}
                        )
                    
                    # Extract intermediate steps if available
                    if "intermediate_steps" in result:
                        intermediate_steps = result["intermediate_steps"]
                        self.logger.info(
                            f"✅ Extracted {len(intermediate_steps)} intermediate steps",
                            extra={"session_id": current_session_id, "steps_count": len(intermediate_steps)}
                        )
                    # Some versions may return "messages" key
                    elif "messages" in result:
                        messages = result["messages"]
                        self.logger.debug(
                            f"Found {len(messages)} messages in result",
                            extra={"session_id": current_session_id}
                        )
                        
                        # Get the last message which should be the final response
                        for idx, msg in enumerate(reversed(messages)):
                            if hasattr(msg, "content") and msg.content:
                                final_output = msg.content
                                self.logger.info(
                                    f"✅ Extracted final output from message {len(messages)-idx-1}: {len(final_output)} chars",
                                    extra={"session_id": current_session_id, "message_type": type(msg).__name__}
                                )
                                break
                            elif isinstance(msg, dict) and "content" in msg:
                                final_output = msg["content"]
                                self.logger.info(
                                    f"✅ Extracted final output from message dict {len(messages)-idx-1}: {len(final_output)} chars",
                                    extra={"session_id": current_session_id}
                                )
                                break
                        
                        # Log tool calls found in messages (already published via on_tool_start)
                        for msg in messages:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                self.logger.info(
                                    f"🔧 Found tool_calls in message: {len(msg.tool_calls)} calls",
                                    extra={"session_id": current_session_id, "tool_calls": msg.tool_calls}
                                )
                    else:
                        # Result is not in expected format, try to extract text from various possible keys
                        # Check common keys that might contain the response text
                        possible_keys = ["result", "response", "text", "content", "message"]
                        extracted = None
                        for key in possible_keys:
                            if key in result and isinstance(result[key], str):
                                extracted = result[key]
                                break
                        
                        if extracted:
                            final_output = extracted
                            self.logger.info(
                                f"✅ Extracted final output from result['{key}']: {len(final_output)} chars",
                                extra={"session_id": current_session_id}
                            )
                        else:
                            # Last resort: try to find any string value in the dict
                            for value in result.values():
                                if isinstance(value, str) and value:
                                    final_output = value
                                    self.logger.warning(
                                        f"Extracted text from unexpected key in result: {type(result)}",
                                        extra={"session_id": current_session_id}
                                    )
                                    break
                            
                            # If still no text found, use accumulated_content or empty string
                            if not final_output:
                                final_output = accumulated_content or ""
                                self.logger.warning(
                                    f"Could not extract text from result, using accumulated_content: {type(result)}",
                                    extra={"session_id": current_session_id, "result_keys": list(result.keys())}
                                )
                else:
                    # Result is not a dict, try to extract text from it
                    if hasattr(result, "content"):
                        final_output = str(result.content) if result.content else ""
                    elif hasattr(result, "text"):
                        final_output = str(result.text) if result.text else ""
                    else:
                        # Last resort: convert to string, but try to avoid dict representation
                        result_str = str(result)
                        # If it looks like a dict representation, try to extract content
                        if result_str.startswith("{") and "'output'" in result_str:
                            # Try to extract from dict-like string (shouldn't happen, but handle it)
                            import re
                            match = re.search(r"'output':\s*['\"]([^'\"]+)['\"]", result_str)
                            if match:
                                final_output = match.group(1)
                            else:
                                final_output = result_str
                        else:
                            final_output = result_str
                    
                    self.logger.warning(
                        f"Result is not a dict, extracted text: {type(result)}",
                        extra={"session_id": current_session_id}
                    )
                
                # Publish final response if we have it (only if session still exists)
                if final_output:
                    await self._safe_publish(EventType.LLM_RESPONSE, {
                        "session_id": current_session_id,
                        "content": final_output,
                        "streaming": False,
                    }, current_session_id)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Log error during agent execution
                # According to LangChain best practices, use msg variable for error messages
                msg = f"Error during AgentExecutor execution: {str(e)}"
                self.logger.error(
                    msg,
                    extra={"session_id": current_session_id},
                    exc_info=True
                )
                
                # Try to extract partial result if available
                try:
                    # Some errors may still have partial results
                    if hasattr(e, "last_result") and e.last_result:
                        result = e.last_result
                        if isinstance(result, dict) and "output" in result:
                            final_output = result["output"]
                            self.logger.info(
                                f"✅ Extracted partial output from error result: {len(final_output) if final_output else 0} chars",
                                extra={"session_id": current_session_id}
                            )
                except Exception:
                    pass
                
                # If no output was extracted, use error message
                if not final_output:
                    final_output = f"Error: {str(e)}"
            
            # Use accumulated content if available (from streaming)
            if accumulated_content and not final_output:
                final_output = accumulated_content
            
            # Note: message_history is per-session and not persisted across executions
            # The storage system handles persistence of messages
            
            # Get reasoning and debug summary from callback
            reasoning_debug_summary = reasoning_callback.get_summary()
            
            duration = time.time() - start_time
            log_event(
                self.logger,
                "tool_calling_executor.completed",
                session_id=current_session_id,
                duration_ms=duration * 1000,
                success=True,
                response_length=len(final_output) if final_output else 0,
                reasoning_steps=len(reasoning_debug_summary["reasoning_steps"]),
                tool_calls=len(reasoning_debug_summary["tool_calls"]),
                tool_errors=len(reasoning_debug_summary["tool_errors"]),
            )
            
            # Build comprehensive result with reasoning, debug info, and steps
            return {
                "success": True,
                "response": final_output or "",
                "messages": [],  # Memory handles message history internally
                # Reasoning and thinking
                "reasoning": {
                    "steps": reasoning_debug_summary["reasoning_steps"],
                    "count": len(reasoning_debug_summary["reasoning_steps"]),
                },
                # Debug information
                "debug": {
                    "llm_calls": reasoning_debug_summary["llm_calls"],
                    "debug_info": reasoning_debug_summary["debug_info"],
                    "summary": reasoning_debug_summary["summary"],
                },
                # Intermediate steps (from AgentExecutor)
                "intermediate_steps": intermediate_steps,
                # Tool execution details
                "tools": {
                    "calls": reasoning_debug_summary["tool_calls"],
                    "errors": reasoning_debug_summary["tool_errors"],
                    "calls_count": len(reasoning_debug_summary["tool_calls"]),
                    "errors_count": len(reasoning_debug_summary["tool_errors"]),
                },
                # Auto-correction info
                "auto_correction": {
                    "enabled": True,  # handle_parsing_errors=True enables auto-correction
                    "errors_encountered": len(reasoning_debug_summary["tool_errors"]),
                    "retries_detected": len([e for e in reasoning_debug_summary["tool_errors"] if e.get("error")]),
                },
            }

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Log execution error
            msg = str(e)
            duration = time.time() - start_time
            log_event(
                self.logger,
                "tool_calling_executor.error",
                session_id=current_session_id,
                duration_ms=duration * 1000,
                error=msg,
                level="ERROR",
            )
            return {
                "success": False,
                "error": f"Execution error: {msg}",
            }


# Per-session executor cache: session_id -> LangChainExecutor
# Each session gets its own executor so model selection is isolated per session.
_session_executors: Dict[str, LangChainExecutor] = {}


def has_session_executor(session_id: str) -> bool:
    """Return True if the session already has a dedicated executor."""
    return session_id in _session_executors


def clear_session_executor(session_id: str) -> None:
    """Remove the executor for a specific session (call when session is deleted)."""
    if session_id in _session_executors:
        _session_executors.pop(session_id, None)
        logger = get_logger("langchain_executor", "executor")
        logger.info(f"Cleared executor for session {session_id}")


def clear_shared_executor() -> None:
    """Clear all session executors (e.g. after a global config/model change).

    Named 'clear_shared_executor' for backward compatibility with callers in
    api/routes/config.py that trigger this on LLM provider switches.
    """
    count = len(_session_executors)
    _session_executors.clear()
    if count:
        logger = get_logger("langchain_executor", "executor")
        logger.info(f"Cleared {count} session executor(s) due to global config change")


async def get_session_executor(
    session_id: str,
    config: AgentConfig,
    tool_registry: ToolRegistry,
    storage: Optional[Storage] = None,
) -> LangChainExecutor:
    """Get or create a dedicated LangChainExecutor for a session.

    The executor is reused across messages within a session as long as the model
    stays the same. When the router selects a different model (e.g. first message
    was a greeting → nano, next is a coding task → smart), a new executor is
    created for that session so the right model handles the task. Conversation
    history is preserved in the DB regardless of executor swaps.

    Args:
        session_id: Session identifier
        config: Agent configuration for this message (may differ from previous)
        tool_registry: Tool registry
        storage: Optional Storage instance for session existence checks

    Returns:
        LangChainExecutor for this session (possibly newly created if model changed)
    """
    _log = get_logger("langchain_executor", "executor")
    existing = _session_executors.get(session_id)

    if existing is not None:
        if existing.config.llm.model == config.llm.model:
            return existing  # Same model — reuse executor
        # Model changed between messages — create new executor; history in DB is preserved
        _log.info(
            f"Session {session_id}: model changed "
            f"{existing.config.llm.model} → {config.llm.model}, recreating executor"
        )

    executor = LangChainExecutor(
        config=config,
        tool_registry=tool_registry,
        storage=storage,
    )
    _session_executors[session_id] = executor
    _log.info(
        f"Created executor for session {session_id} "
        f"(model={config.llm.model}, total_sessions={len(_session_executors)})"
    )
    return executor



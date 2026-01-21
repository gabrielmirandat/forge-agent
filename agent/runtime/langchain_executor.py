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

import os
import time
from typing import Any, Dict, List, Optional

from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Memory modules removed - not available in LangChain 1.2.6
# Will use manual message history management instead

from agent.config.loader import AgentConfig
from agent.observability import get_logger, log_event
from agent.runtime.bus import EventType, publish
from agent.runtime.callbacks import ErrorHandlingCallbackHandler
from agent.tools.base import ToolRegistry


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
        session_id: Optional[str] = None,
        max_iterations: int = 50,
        buffer_window_size: int = 10,
    ) -> None:
        """Initialize Tool-Calling agent executor.
        
        All LLM providers and tools are loaded/executed through LangChain.
        The executor uses create_agent to build a stateful agent graph that
        manages tool calling loops automatically.
        
        Args:
            config: Agent configuration containing LLM settings and runtime options.
            tool_registry: Tool registry containing all available tools.
            session_id: Optional session ID for tracking and state management.
            max_iterations: Maximum number of agent iterations before stopping.
            buffer_window_size: Number of recent messages to keep in context buffer.
        """
        self.config = config
        self.tool_registry = tool_registry
        self.session_id = session_id
        self.max_iterations = max_iterations
        self.buffer_window_size = buffer_window_size
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
        
        # Create ChatOllama instance
        # Note: Docker Manager ensures Ollama container is running before this point
        # The base_url should point to the Docker Ollama container (default: http://localhost:11434)
        # Reference: https://docs.ollama.com/capabilities/tool-calling#python
        # Ollama supports tool calling natively - LangChain's bind_tools() will convert tools to Ollama format
        # IMPORTANT: Not all Ollama models support tool calling!
        # Models that support tools: qwen3:8b, devstral, granite4, command-r, etc.
        # Check: https://ollama.com/search?c=tools
        # Note: llama3.1 removed - tool calling does not work reliably
        self.langchain_llm = ChatOllama(
            model=config.llm.model,
            base_url=config.llm.base_url or "http://localhost:11434",
            temperature=config.llm.temperature,
            num_predict=config.llm.max_tokens,  # Ollama uses num_predict instead of max_tokens
            timeout=config.llm.timeout,
            # Note: Ollama tool calling format is handled automatically by LangChain's bind_tools()
            # The format expected by Ollama API is: {"type": "function", "function": {"name": "...", "arguments": {...}}}
            # LangChain converts this automatically when using bind_tools()
        )
        
        # Log connection info for debugging
        self.logger.info(
            f"ChatOllama initialized: model={config.llm.model}, "
            f"base_url={config.llm.base_url or 'http://localhost:11434'}"
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
        self.message_history: List[Dict[str, Any]] = []
        self.buffer_window_size = buffer_window_size
        
        # Create error handling callback
        # Pass model name to callback for metrics tracking
        model_name = config.llm.model if config.llm else None
        self.callback_handler = ErrorHandlingCallbackHandler(
            session_id=session_id,
            model_name=model_name
        )
        
        # Get LangSmith callbacks if tracing is enabled
        langsmith_callbacks = self._get_langsmith_callbacks()
        
        # Combine callbacks: error handling + LangSmith tracing
        self.all_callbacks = [self.callback_handler]
        if langsmith_callbacks:
            self.all_callbacks.extend(langsmith_callbacks)

    async def _ensure_agent_initialized(self) -> None:
        """Initialize the agent graph and bind tools to the model.
        
        This method is called lazily on first use to avoid event loop issues.
        It performs the following steps:
        1. Converts tools from registry to LangChain format
        2. Binds tools to the LLM using bind_tools()
        3. Builds the system prompt
        4. Creates the agent graph using create_agent()
        
        Raises:
            ImportError: If required LangChain modules are not available.
            ValueError: If tool binding fails or agent creation fails.
        """
        """Lazily load tools and create the LangChain agent graph in an async context.
        
        This method must be called from async code (e.g., run()) to avoid
        manipulating the event loop (uvloop) in __init__.
        """
        if self.agent_executor is not None:
            return

        # Load LangChain tools from MCP servers using langchain-mcp-adapters (official pattern)
        self.langchain_tools = await self.tool_registry.get_langchain_tools(
            session_id=self.session_id,
            config=self.config,
        )

        # Log tools for debugging
        self.logger.info(
            f"Initialized with {len(self.langchain_tools)} LangChain tools",
            extra={"tool_names": [tool.name for tool in self.langchain_tools[:10]]},  # First 10 tools
        )

        # Bind tools to the model using bind_tools() (LangChain best practice)
        # References:
        # - LangChain: https://docs.langchain.com/oss/python/langchain/models#example-nested-structures
        # - Ollama: https://docs.ollama.com/capabilities/tool-calling#python
        # When bind_tools() is used, LangChain automatically converts tools to Ollama's format:
        # {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        # Ollama returns tool_calls in format: {"type": "function", "function": {"name": "...", "arguments": {...}}}
        # We can use tool_choice parameter to control tool usage: "auto", "required", or specific tool name
        try:
            # Check if config has tool_choice setting (for forcing tool usage)
            # tool_choice can be set in config.llm.tool_choice
            tool_choice = getattr(self.config.llm, "tool_choice", None)
            if tool_choice is None:
                # Default: "auto" - let model decide when to use tools
                tool_choice = "auto"
            
            # Bind tools with optional tool_choice
            # According to LangChain tests, tool_choice can be: "auto", "required", "any", or a tool name
            # Reference: _helpers/langchain/libs/standard-tests/langchain_tests/unit_tests/chat_models.py
            if tool_choice != "auto":
                # Only add tool_choice if it's explicitly set (some models may not support it)
                try:
                    # Try to use tool_choice if the model supports it
                    self.langchain_llm_with_tools = self.langchain_llm.bind_tools(
                        self.langchain_tools,
                        tool_choice=tool_choice
                    )
                    self.logger.info(
                        f"✅ Tools bound with tool_choice='{tool_choice}' - {len(self.langchain_tools)} tools",
                        extra={"session_id": self.session_id, "tool_choice": tool_choice},
                    )
                except TypeError:
                    # Model doesn't support tool_choice parameter, fall back to default
                    self.logger.debug(
                        f"Model doesn't support tool_choice parameter, using default bind_tools()",
                        extra={"session_id": self.session_id},
                    )
                    self.langchain_llm_with_tools = self.langchain_llm.bind_tools(self.langchain_tools)
            else:
                # Default: just bind tools without tool_choice
                self.langchain_llm_with_tools = self.langchain_llm.bind_tools(self.langchain_tools)
            
            # Log tool names for debugging
            tool_names = [tool.name for tool in self.langchain_tools[:20]]  # First 20
            self.logger.info(
                f"✅ Tools bound to model using bind_tools() - {len(self.langchain_tools)} tools",
                extra={
                    "session_id": self.session_id,
                    "tools_count": len(self.langchain_tools),
                    "tool_names_sample": tool_names,
                    "tool_choice": tool_choice,
                },
            )
            
            # Verify bind_tools worked by checking if model has tool calling capability
            if hasattr(self.langchain_llm_with_tools, "bound_tools"):
                bound_tools_count = len(self.langchain_llm_with_tools.bound_tools) if self.langchain_llm_with_tools.bound_tools else 0
                self.logger.info(
                    f"✅ Verified: Model has {bound_tools_count} bound tools",
                    extra={"session_id": self.session_id},
                )
        except Exception as e:
            # Log error but continue without tool binding
            # This allows the agent to work even if tool binding fails
            # According to LangChain best practices, use msg variable for error messages
            msg = f"bind_tools() failed: {str(e)}"
            self.logger.warning(
                f"⚠️ {msg}, using model without bind_tools",
                extra={"session_id": self.session_id},
                exc_info=True,
            )
            self.langchain_llm_with_tools = self.langchain_llm

        # Build system prompt for the agent (needs langchain_tools to be set first)
        system_prompt_str = await self._format_system_prompt()

        self.logger.info(
            "System prompt built",
            extra={
                "session_id": self.session_id,
                "prompt_length": len(system_prompt_str),
                "tools_count": len(self.langchain_tools),
            }
        )

        # Create ChatPromptTemplate for create_tool_calling_agent
        # According to LangChain documentation, the prompt must have:
        # - "agent_scratchpad" as MessagesPlaceholder (required)
        # - "chat_history" as MessagesPlaceholder (optional, for conversation history)
        # - "input" for user input
        # Reference: _helpers/langchain/libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt_str),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        # Create agent using create_tool_calling_agent
        # According to LangChain documentation:
        # - llm: LLM with tools bound via bind_tools() (already done above)
        # - tools: List of tools for the agent to execute
        # - prompt: ChatPromptTemplate with agent_scratchpad MessagesPlaceholder
        # Reference: _helpers/langchain/libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
        try:
            self.agent = create_tool_calling_agent(
                llm=self.langchain_llm_with_tools,  # Model with tools already bound via bind_tools()
                tools=self.langchain_tools,  # Tools list
                prompt=self.prompt_template,  # ChatPromptTemplate with required placeholders
            )
            
            self.logger.info(
                "✅ Agent created with create_tool_calling_agent",
                extra={"session_id": self.session_id}
            )
        except Exception as e:
            # Log error and re-raise - this is a critical failure
            msg = f"Failed to create tool calling agent: {str(e)}"
            self.logger.error(
                f"❌ {msg}",
                extra={"session_id": self.session_id},
                exc_info=True,
            )
            raise ValueError(msg) from e
        
        # Create AgentExecutor to run the agent
        # AgentExecutor manages the tool calling loop and handles errors
        # Reference: _helpers/langchain/libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
        try:
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.langchain_tools,
                verbose=True,  # Enable verbose logging for debugging
                max_iterations=self.max_iterations,  # Limit iterations to prevent infinite loops
                handle_parsing_errors=True,  # Handle tool call parsing errors gracefully
            )
            
            self.logger.info(
                f"✅ AgentExecutor created with {len(self.langchain_tools)} tools",
                extra={
                    "session_id": self.session_id,
                    "tools_count": len(self.langchain_tools),
                    "model_type": type(self.langchain_llm_with_tools).__name__,
                    "max_iterations": self.max_iterations,
                }
            )
        except Exception as e:
            # Log error and re-raise - this is a critical failure
            msg = f"Failed to create AgentExecutor: {str(e)}"
            self.logger.error(
                f"❌ {msg}",
                extra={"session_id": self.session_id},
                exc_info=True,
            )
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
                # Use session_id or default project name
                project_name = f"forge-agent-{self.session_id[:8]}" if self.session_id else "forge-agent"
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
    
    def _get_langsmith_callbacks(self) -> List[Any]:
        """Get LangSmith callbacks if tracing is enabled.
        
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
            if self.session_id:
                # Set run name for better trace identification
                tracer.run_name = f"agent-execution-{self.session_id[:8]}"
            
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
    def _generate_server_description(server_name: str, tools: List[str]) -> str:
        """Generate a description for an MCP server based on its name and tools.
        
        Args:
            server_name: Name of the MCP server
            tools: List of tool names from this server
            
        Returns:
            Description string
        """
        # Generate description based on server name and tools
        server_lower = server_name.lower()
        
        # Common server descriptions
        descriptions = {
            "desktop_commander": "Handle filesystem and shell operations",
            "playwright": "Handle browser e2e tests",
            "git": "Handle git operations",
            "github": "Handle GitHub API operations",
            "openapi": "Handle OpenAPI/Swagger API operations",
            "python_refactoring": "Handle Python code refactoring",
            "fetch": "Handle web content fetching",
        }
        
        # Try exact match first
        if server_name in descriptions:
            return descriptions[server_name]
        
        # Try lowercase match
        if server_lower in descriptions:
            return descriptions[server_lower]
        
        # Generate from server name
        if "git" in server_lower:
            return "Handle git operations"
        elif "browser" in server_lower or "playwright" in server_lower:
            return "Handle browser automation"
        elif "file" in server_lower or "fs" in server_lower or "commander" in server_lower:
            return "Handle filesystem operations"
        elif "api" in server_lower or "openapi" in server_lower:
            return "Handle API operations"
        elif "refactor" in server_lower or "python" in server_lower:
            return "Handle code refactoring"
        elif "fetch" in server_lower or "web" in server_lower:
            return "Handle web content operations"
        else:
            # Generic description
            return f"Handle {server_name.replace('_', ' ')} operations"
    
    @staticmethod
    async def format_system_prompt(
        config: Any,
        langchain_tools: List[Any]
    ) -> str:
        """Format system prompt with tool information.
        
        This static method can be called from outside to preview the system prompt
        that will be sent to the LLM.
        
        Available template variables:
        - {workspace_base}: Workspace base path
        - {tools}: List of MCP servers with their tools in format "tool : server_name - description - tool1, tool2, ..."
        
        Args:
            config: AgentConfig instance
            langchain_tools: List of LangChain tools (not used directly, we query MCP manager instead)
            
        Returns:
            Formatted system prompt string
        """
        workspace_base = "~/repos"
        if hasattr(config, "workspace") and hasattr(config.workspace, "base_path"):
            workspace_base = config.workspace.base_path
        
        # Get tools directly from MCP manager (grouped by server)
        # This is more reliable than trying to extract server names from tool names
        from agent.runtime.mcp_client import get_mcp_manager
        
        mcp_manager = get_mcp_manager()
        all_tools_by_server = mcp_manager.get_all_tools()
        
        # Format tools: "tool : server_name - description - tool1, tool2, ..."
        tools_list = []
        for server_name in sorted(all_tools_by_server.keys()):
            mcp_tools = all_tools_by_server[server_name]
            tool_names = [tool.name for tool in mcp_tools]
            tools_str = ", ".join(sorted(tool_names))
            description = LangChainExecutor._generate_server_description(server_name, tool_names)
            tools_list.append(f"tool : {server_name} - {description} - {tools_str}")
        
        tools_str = "\n".join(tools_list) if tools_list else "No tools available"
        
        # Use custom template from config if available, otherwise use default
        if hasattr(config, "system_prompt_template") and config.system_prompt_template:
            template = config.system_prompt_template
        else:
            # Default template
            template = """You are a helpful coding assistant with access to tools.

Workspace: {workspace_base}

Available tools:
{tools}

Use tools when the user asks you to perform actions. For simple questions, answer directly.

The tool schemas and parameters are provided to you automatically - use them as needed."""
        
        # Format template with placeholders
        # Available variables:
        # - {workspace_base}: Workspace base path
        # - {tools}: MCP servers with tools (format: "tool : server_name - description - tool1, tool2, ...")
        formatted = template.format(
            workspace_base=workspace_base,
            tools=tools_str
        )
        
        return formatted
    
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
            
        Returns:
            Dictionary with final response and execution history
        """
        start_time = time.time()
        log_event(
            self.logger,
            "tool_calling_executor.started",
            session_id=self.session_id,
            max_iterations=self.max_iterations,
        )

        # Ensure tools and agent graph are initialized in this async context
        await self._ensure_agent_initialized()
        try:
            # Initialize memory from conversation history if provided
            # Only initialize if message history is empty (avoid duplicates)
            if conversation_history and len(self.message_history) == 0:
                # Process messages and add to history
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"]:
                        self.message_history.append({
                            "role": role,
                            "content": content,
                        })
            
            # Load memory variables (recent messages)
            memory_vars = self._load_memory_variables()
            chat_history = memory_vars.get("chat_history", [])
            
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
            input_data: Dict[str, Any] = {
                "input": user_message,
            }
            
            # Add chat history if available
            if langchain_chat_history:
                input_data["chat_history"] = langchain_chat_history
            
            log_event(
                self.logger,
                "tool_calling_executor.agent.invoking",
                session_id=self.session_id,
                buffer_messages=len(chat_history),
            )
            
            # Log available tools for debugging
            self.logger.info(
                f"Available tools for agent: {[t.name for t in self.langchain_tools[:10]]}",
                extra={"session_id": self.session_id, "total_tools": len(self.langchain_tools)}
            )
            self.logger.info(
                f"User message: {user_message[:200]}",
                extra={"session_id": self.session_id}
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
                    extra={"session_id": self.session_id, "prompt_length": len(system_prompt_for_log)}
                )
            
            # Use AgentExecutor to run the agent
            # AgentExecutor manages the tool calling loop automatically
            # Reference: _helpers/langchain/libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
            final_output = None
            accumulated_content = ""
            
            # Prepare config for agent invocation (callbacks for observability)
            agent_config: Dict[str, Any] = {"callbacks": self.all_callbacks}
            
            try:
                # Invoke AgentExecutor (synchronous invoke, but we're in async context)
                # AgentExecutor.invoke() is synchronous, so we run it in executor
                import asyncio
                result = await asyncio.to_thread(
                    self.agent_executor.invoke,
                    input_data,
                    config=agent_config,
                )
                
                self.logger.debug(
                    f"AgentExecutor result type={type(result)}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}",
                    extra={"session_id": self.session_id}
                )
                
                # Extract final output from AgentExecutor result
                # AgentExecutor returns a dict with "output" key containing the final response
                # Reference: _helpers/langchain/libs/langchain/langchain_classic/agents/tool_calling_agent/base.py
                if isinstance(result, dict):
                    # AgentExecutor typically returns {"output": "..."}
                    if "output" in result:
                        final_output = result["output"]
                        self.logger.info(
                            f"✅ Extracted final output from AgentExecutor result['output']: {len(final_output) if final_output else 0} chars",
                            extra={"session_id": self.session_id}
                        )
                    # Some versions may return "messages" key
                    elif "messages" in result:
                        messages = result["messages"]
                        self.logger.debug(
                            f"Found {len(messages)} messages in result",
                            extra={"session_id": self.session_id}
                        )
                        
                        # Get the last message which should be the final response
                        for idx, msg in enumerate(reversed(messages)):
                            if hasattr(msg, "content") and msg.content:
                                final_output = msg.content
                                self.logger.info(
                                    f"✅ Extracted final output from message {len(messages)-idx-1}: {len(final_output)} chars",
                                    extra={"session_id": self.session_id, "message_type": type(msg).__name__}
                                )
                                break
                            elif isinstance(msg, dict) and "content" in msg:
                                final_output = msg["content"]
                                self.logger.info(
                                    f"✅ Extracted final output from message dict {len(messages)-idx-1}: {len(final_output)} chars",
                                    extra={"session_id": self.session_id}
                                )
                                break
                        
                        # Also check for tool calls in messages for logging
                        for msg in messages:
                            if hasattr(msg, "tool_calls") and msg.tool_calls:
                                self.logger.info(
                                    f"🔧 Found tool_calls in message: {len(msg.tool_calls)} calls",
                                    extra={"session_id": self.session_id, "tool_calls": msg.tool_calls}
                                )
                                # Extract tool call info (LangChain format)
                                tool_call = msg.tool_calls[0]
                                tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else getattr(tool_call, "name", "")
                                tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                                
                                await publish(EventType.TOOL_CALLED, {
                                    "session_id": self.session_id,
                                    "tool": tool_name,
                                    "arguments": tool_args,
                                })
                    else:
                        # Result is not in expected format, try to extract as string
                        final_output = str(result.get("result", result))
                        self.logger.warning(
                            f"Result is not in expected format, using result/string: {type(result)}",
                            extra={"session_id": self.session_id}
                        )
                else:
                    # Result is not a dict, convert to string
                    final_output = str(result)
                    self.logger.warning(
                        f"Result is not a dict, converting to string: {type(result)}",
                        extra={"session_id": self.session_id}
                    )
                
                # Publish final response if we have it
                if final_output:
                    await publish(EventType.LLM_RESPONSE, {
                        "session_id": self.session_id,
                        "content": final_output,
                        "streaming": False,
                    })
                    
            except Exception as e:
                # Log error during agent execution
                # According to LangChain best practices, use msg variable for error messages
                msg = f"Error during AgentExecutor execution: {str(e)}"
                self.logger.error(
                    msg,
                    extra={"session_id": self.session_id},
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
                                extra={"session_id": self.session_id}
                            )
                except Exception:
                    pass
                
                # If no output was extracted, use error message
                if not final_output:
                    final_output = f"Error: {str(e)}"
            
            # Use accumulated content if available (from streaming)
            if accumulated_content and not final_output:
                final_output = accumulated_content
            
            # Save conversation to memory (for next interaction)
            self._save_context(user_message, final_output or "")
            
            duration = time.time() - start_time
            log_event(
                self.logger,
                "tool_calling_executor.completed",
                session_id=self.session_id,
                duration_ms=duration * 1000,
                success=True,
                response_length=len(final_output) if final_output else 0,
            )
            
            return {
                "success": True,
                "response": final_output or "",
                "messages": [],  # Memory handles message history internally
            }
                
        except Exception as e:
            # Log execution error
            # According to LangChain best practices, use msg variable for error messages
            msg = str(e)
            duration = time.time() - start_time
            log_event(
                self.logger,
                "tool_calling_executor.error",
                session_id=self.session_id,
                duration_ms=duration * 1000,
                error=msg,
                level="ERROR",
            )
            return {
                "success": False,
                "error": f"Execution error: {msg}",
            }

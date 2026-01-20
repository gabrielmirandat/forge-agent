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

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
# Memory modules removed - not available in LangChain 1.2.6
# Will use manual message history management instead

from agent.config.loader import AgentConfig
from agent.observability import get_logger, log_event
from agent.runtime.bus import EventType, publish
from agent.runtime.callbacks import ErrorHandlingCallbackHandler
from agent.tools.base import ToolRegistry


class LangChainExecutor:
    """Executor using Tool-Calling agents with LangChain.
    
    Uses LangChain's create_agent (LangChain 1.2.6+):
    - create_agent creates a CompiledStateGraph that can invoke tools
    - Agent graph manages the loop: model → tool calls → tool results → model
    - LLM directly calls tools via function calling (native model support)
    - More efficient than ReAct (no explicit Thought/Action/Observation loop)
    - Better for production use
    - Callbacks provide observability and error handling
    """
    
    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        session_id: Optional[str] = None,
        max_iterations: int = 50,
        buffer_window_size: int = 10,
    ):
        """Initialize Tool-Calling agent executor.
        
        All LLM providers and tools are loaded/executed through LangChain.
        
        Args:
            config: Agent configuration
            tool_registry: Tool registry
            session_id: Optional session ID
            max_iterations: Maximum number of iterations (default: 50)
            buffer_window_size: Number of recent messages to keep in buffer (default: 10)
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
        
        # Tools and agent graph are loaded lazily in the async context (run/_ensure_agent_initialized)
        # This avoids trying to patch or manage the event loop (e.g., uvloop) in __init__,
        # which caused errors when running inside FastAPI/uvicorn.
        self.langchain_tools: List[Any] = []
        self.langchain_llm_with_tools = self.langchain_llm
        self.system_prompt_template = None
        self.agent_graph = None
        self._checkpointer = None  # Store checkpointer for agent config
        
        # Memory management: manual message history
        # LangChain 1.2.6 does not provide ConversationBufferWindowMemory/ConversationSummaryMemory
        # in the same way, so we manage message history manually.
        self.message_history: List[Dict[str, Any]] = []
        self.buffer_window_size = buffer_window_size
        
        # Create error handling callback
        self.callback_handler = ErrorHandlingCallbackHandler(session_id=session_id)
        
        # Get LangSmith callbacks if tracing is enabled
        langsmith_callbacks = self._get_langsmith_callbacks()
        
        # Combine callbacks: error handling + LangSmith tracing
        self.all_callbacks = [self.callback_handler]
        if langsmith_callbacks:
            self.all_callbacks.extend(langsmith_callbacks)

    async def _ensure_agent_initialized(self) -> None:
        """Lazily load tools and create the LangChain agent graph in an async context.
        
        This method must be called from async code (e.g., run()) to avoid
        manipulating the event loop (uvloop) in __init__.
        """
        if self.agent_graph is not None:
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
            # tool_choice can be: "auto", "required", or a specific tool name
            bind_kwargs = {}
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
            self.logger.warning(
                f"⚠️ bind_tools() failed ({e}), using model without bind_tools",
                extra={"session_id": self.session_id},
                exc_info=True,
            )
            self.langchain_llm_with_tools = self.langchain_llm

        # Build system prompt for the agent (needs langchain_tools to be set first)
        self.system_prompt_template = await self._build_system_prompt()

        # Format system prompt to string (create_agent expects string, not ChatPromptTemplate)
        system_messages = self.system_prompt_template.format_messages()
        system_prompt_str = ""
        for msg in system_messages:
            if hasattr(msg, "content"):
                system_prompt_str += msg.content + "\n"
        system_prompt_str = system_prompt_str.strip()

        self.logger.info(
            "System prompt built",
            extra={
                "session_id": self.session_id,
                "prompt_length": len(system_prompt_str),
                "tools_count": len(self.langchain_tools),
            }
        )

        # Create agent using LangChain 1.2.6+ API
        # Based on langchain-mcp-adapters tests, create_agent should work correctly
        # when model has tools bound and proper state schema is used
        try:
            # Try to use create_agent with state_schema and checkpointer (recommended pattern)
            from langchain.agents import AgentState
            from langgraph.checkpoint.memory import MemorySaver
            
            # Create checkpointer for state management
            self._checkpointer = MemorySaver()
            
            # Create agent with state schema and checkpointer
            # Based on LangChain docs: https://docs.langchain.com/oss/python/langchain/models#example-nested-structures
            # When using bind_tools(), the model automatically receives tool schemas via function calling
            # create_agent manages the tool calling loop: model → tool calls → tool results → model
            self.agent_graph = create_agent(
                model=self.langchain_llm_with_tools,  # Model with tools already bound via bind_tools()
                tools=self.langchain_tools,  # Tools list (also passed for agent to execute)
                system_prompt=system_prompt_str,
                state_schema=AgentState,
                checkpointer=self._checkpointer,
            )
            
            self.logger.info(
                "✅ Agent created with state_schema and checkpointer",
                extra={"session_id": self.session_id}
            )
        except (ImportError, TypeError) as e:
            # Fallback to basic create_agent if state_schema/checkpointer not available
            self.logger.debug(
                f"State schema/checkpointer not available, using basic create_agent: {e}",
                extra={"session_id": self.session_id}
            )
            self.agent_graph = create_agent(
                model=self.langchain_llm_with_tools,
                tools=self.langchain_tools,
                system_prompt=system_prompt_str,
            )
        
        # Log agent creation for debugging
        self.logger.info(
            f"✅ Agent graph created with {len(self.langchain_tools)} tools",
            extra={
                "session_id": self.session_id,
                "tools_count": len(self.langchain_tools),
                "model_type": type(self.langchain_llm_with_tools).__name__,
            }
        )
    
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
            self.logger.warning(f"Failed to initialize LangSmith tracer: {e}")
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
    
    async def _build_system_prompt(self):
        """Build system prompt for the agent using ChatPromptTemplate.
        
        Returns:
            ChatPromptTemplate instance
            
        Note: According to LangChain docs (https://docs.langchain.com/oss/python/langchain/models#example-nested-structures),
        when using bind_tools(), tool schemas are provided automatically via function calling.
        The LLM receives tool definitions automatically, so we focus the prompt on WHEN to use tools,
        not on listing all tool details.
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        # Format the prompt using the same logic
        system_prompt_str = await self._format_system_prompt()
        
        # Create ChatPromptTemplate from formatted string
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_str),
        ])
        
        return prompt
    
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
            
            # Build messages for agent with memory context
            # Start with system prompt messages
            system_messages = self.system_prompt_template.format_messages()
            messages = list(system_messages)
            
            # Convert chat history to LangChain messages
            for msg_dict in chat_history:
                role = msg_dict.get("role", "")
                content = msg_dict.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
            
            # Add current user message
            messages.append(HumanMessage(content=user_message))
            
            # Build input for agent graph
            # For tool-calling agents, we pass messages directly
            input_data = {"messages": messages}
            
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
            
            # Log system prompt being used (extract from template)
            system_messages_for_log = self.system_prompt_template.format_messages()
            system_prompt_for_log = ""
            for msg in system_messages_for_log:
                if hasattr(msg, "content"):
                    system_prompt_for_log += msg.content + "\n"
            system_prompt_for_log = system_prompt_for_log.strip()
            system_prompt_debug = system_prompt_for_log[:500] if system_prompt_for_log else "No system prompt"
            self.logger.info(
                f"System prompt (first 500 chars): {system_prompt_debug}",
                extra={"session_id": self.session_id, "prompt_length": len(system_prompt_for_log)}
            )
            
            # Use create_agent with proper configuration (based on langchain-mcp-adapters tests)
            # The agent graph manages the tool calling loop automatically
            final_output = None
            accumulated_content = ""
            
            # Prepare config for agent invocation
            # If using checkpointer, we need to provide thread_id
            agent_config = {"callbacks": self.all_callbacks}
            try:
                from langchain.agents import AgentState
                from langgraph.checkpoint.memory import MemorySaver
                # If checkpointer is used, add thread_id to config
                if hasattr(self, '_checkpointer') and self._checkpointer:
                    agent_config["configurable"] = {"thread_id": self.session_id or "default"}
            except ImportError:
                pass
            
            # Try astream_events first for observability, but also invoke directly to get result
            # Based on langchain-mcp-adapters tests, we should use ainvoke and extract from messages
            try:
                # Invoke agent directly (this is the recommended pattern from tests)
                result = await self.agent_graph.ainvoke(
                    input_data,
                    config=agent_config,
                )
                
                self.logger.debug(
                    f"Agent result type={type(result)}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}",
                    extra={"session_id": self.session_id}
                )
                
                # Extract final message content from result
                # Based on langchain-mcp-adapters tests, result should have "messages" key
                if isinstance(result, dict) and "messages" in result:
                    messages = result["messages"]
                    self.logger.debug(
                        f"Found {len(messages)} messages in result",
                        extra={"session_id": self.session_id}
                    )
                    
                    # Get the last message which should be the final response
                    # Check all messages in reverse to find the last one with content
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
                    # Reference: https://docs.ollama.com/capabilities/tool-calling#python
                    # Ollama returns tool_calls in format: {"type": "function", "function": {"name": "...", "arguments": {...}}}
                    # LangChain converts this to: {"name": "...", "args": {...}, "id": "...", "type": "tool_call"}
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
                    # Result is not in expected format
                    final_output = str(result)
                    self.logger.warning(
                        f"Result is not a dict with messages, converting to string: {type(result)}",
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
                # Fallback if something goes wrong
                self.logger.error(
                    f"Error during agent execution: {e}",
                    extra={"session_id": self.session_id},
                    exc_info=True
                )
                
                # Try to invoke directly as last resort
                try:
                    result = await self.agent_graph.ainvoke(
                        input_data,
                        config=agent_config,
                    )
                    
                    if isinstance(result, dict) and "messages" in result:
                        messages = result["messages"]
                        for msg in reversed(messages):
                            if hasattr(msg, "content") and msg.content:
                                final_output = msg.content
                                break
                except Exception as e2:
                    self.logger.error(
                        f"Fallback ainvoke also failed: {e2}",
                        extra={"session_id": self.session_id},
                        exc_info=True
                    )
                    final_output = f"Error: {str(e)}"
                # Fallback to ainvoke if astream_events is not available
                self.logger.warning(
                    f"astream_events not available, falling back to ainvoke: {e}"
                )
                
                # Invoke agent directly
                result = await self.agent_graph.ainvoke(
                    input_data,
                    config=agent_config,
                )
                
                self.logger.debug(
                    f"ainvoke result type={type(result)}, keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}",
                    extra={"session_id": self.session_id}
                )
                
                # Extract final message content
                if isinstance(result, dict):
                    if "messages" in result:
                        messages = result["messages"]
                        self.logger.debug(
                            f"ainvoke: found {len(messages)} messages",
                            extra={"session_id": self.session_id}
                        )
                        
                        if messages:
                            # Get the last message which should be the final response
                            last_msg = messages[-1]
                            self.logger.debug(
                                f"ainvoke: last_msg type={type(last_msg)}, has content={hasattr(last_msg, 'content')}",
                                extra={"session_id": self.session_id}
                            )
                            
                            if hasattr(last_msg, "content"):
                                final_output = last_msg.content
                                self.logger.info(
                                    f"✅ Extracted final output from ainvoke last_msg.content: {len(final_output) if final_output else 0} chars",
                                    extra={"session_id": self.session_id}
                                )
                            elif isinstance(last_msg, dict) and "content" in last_msg:
                                final_output = last_msg["content"]
                            else:
                                # Check all messages for content
                                for idx, msg in enumerate(reversed(messages)):
                                    if hasattr(msg, "content") and msg.content:
                                        final_output = msg.content
                                        self.logger.info(
                                            f"✅ Extracted final output from ainvoke message {len(messages)-idx-1}: {len(final_output)} chars",
                                            extra={"session_id": self.session_id}
                                        )
                                        break
                    else:
                        final_output = str(result)
                else:
                    final_output = str(result)
                
                if final_output:
                    await publish(EventType.LLM_RESPONSE, {
                        "session_id": self.session_id,
                        "content": final_output,
                        "streaming": False,
                    })
                else:
                    self.logger.warning(
                        "No final output extracted from ainvoke result",
                        extra={"session_id": self.session_id, "result_type": type(result)}
                    )
            
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
            duration = time.time() - start_time
            log_event(
                self.logger,
                "tool_calling_executor.error",
                session_id=self.session_id,
                duration_ms=duration * 1000,
                error=str(e),
                level="ERROR",
            )
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
            }

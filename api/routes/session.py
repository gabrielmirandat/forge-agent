"""Session/chat API routes."""

import time
import uuid
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from agent.observability import (
    api_request_duration_seconds,
    api_requests_total,
    get_logger,
    log_event,
    set_request_id,
    set_run_id,
    trace_span,
)
from agent.runtime.executor import Executor
from agent.runtime.planner import Planner
from agent.runtime.schema import (
    ExecutionPolicy,
    InvalidPlanError,
    LLMCommunicationError,
    PlanningError,
    get_operations_requiring_approval,
)
from agent.storage import MessageRole, NotFoundError, Storage, StorageError
from api.dependencies import get_config, get_executor, get_llm_provider, get_planner, get_storage, get_tool_registry
from api.schemas.session import (
    ApproveOperationsRequest,
    ApproveOperationsResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    MessageRequest,
    MessageResponse,
    RejectOperationsRequest,
    SessionResponse,
    SessionsListResponse,
)

router = APIRouter()
logger = get_logger("api.session", "api")


async def _generate_final_response(
    llm_provider,
    user_message: str,
    execution_result,
    config,
) -> str:
    """Generate final response from LLM based on execution results.
    
    Args:
        llm_provider: LLM provider instance
        user_message: Original user message
        execution_result: ExecutionResult object
        config: Agent config
        
    Returns:
        Final formatted response from LLM
    """
    import json
    
    # If no execution result (empty plan), answer directly
    if execution_result is None:
        system_prompt = """You are a helpful assistant. Answer the user's question directly and clearly.

If the question is simple (math, general knowledge, etc.), provide a direct answer without mentioning tools or execution."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        
        try:
            response = await llm_provider.chat(
                messages,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return f"I understand your question, but I encountered an error processing it. Please try again."
    
    # Build execution summary for LLM
    execution_summary = {
        "success": execution_result.success,
        "steps": []
    }
    
    for step in execution_result.steps:
        step_info = {
            "step_id": step.step_id,
            "tool": step.tool,
            "operation": step.operation,
            "success": step.success,
        }
        if step.output is not None:
            # Include output (truncate if too large, but preserve important fields)
            if isinstance(step.output, str):
                step_info["output"] = step.output[:5000]  # Limit to 5000 chars
            elif isinstance(step.output, dict):
                # For dict outputs, preserve important metadata fields even if content is truncated
                output_dict = step.output.copy()
                
                # Always preserve metadata fields (lines, size, path, etc.)
                metadata_fields = ["lines", "size", "path", "return_code", "command"]
                preserved_metadata = {k: v for k, v in output_dict.items() if k in metadata_fields}
                
                # Truncate content if present
                if "content" in output_dict:
                    content = output_dict["content"]
                    if isinstance(content, str) and len(content) > 5000:
                        output_dict["content"] = content[:5000] + f"\n... (truncated, showing first 5000 of {len(content)} characters)"
                
                # Build final output with preserved metadata
                final_output = {**preserved_metadata, **{k: v for k, v in output_dict.items() if k not in metadata_fields}}
                step_info["output"] = json.dumps(final_output, indent=2)
            else:
                step_info["output"] = str(step.output)[:5000]
        if step.error:
            step_info["error"] = step.error
        execution_summary["steps"].append(step_info)
    
    # Build prompt for LLM
    system_prompt = """You are a helpful assistant. Your role is to provide clear, user-friendly responses based on the execution results of commands and operations.

You will receive:
1. The user's original request
2. The execution results (success/failure, outputs, errors)

Your task is to:
- Provide a clear, natural language response to the user
- If the execution result has no steps (empty plan), answer the user's question directly without mentioning tools or execution
- Show the relevant information from the execution results when tools were used
- Format outputs nicely (use code blocks for code, lists for directories, etc.)
- If there were errors, explain them clearly
- Be concise but informative

IMPORTANT: If the execution result has an empty plan (no steps), this means the user's question doesn't require tools. Answer the question directly as a helpful assistant would, without mentioning tools or execution results.

Format your response as plain text with markdown-style formatting (code blocks, lists, etc.)."""

    user_prompt = f"""User request: {user_message}

Execution results:
{json.dumps(execution_summary, indent=2)}

Please provide a clear, user-friendly response based on these execution results."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Call LLM to generate response
    try:
        response = await llm_provider.chat(
            messages,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        return response.strip()
    except Exception as e:
        # Fallback to simple format if LLM fails
        logger.error(f"Failed to generate LLM response: {e}")
        if execution_result is None:
            return f"I understand your question, but I encountered an error processing it. Please try again."
        elif execution_result.success:
            return "✅ Operation completed successfully."
        else:
            return f"❌ Operation failed: {execution_result.steps[-1].error if execution_result.steps else 'Unknown error'}"


def _format_execution_output(execution_result, user_message: str | None = None) -> str:
    """Format execution result into user-friendly output.
    
    Args:
        execution_result: ExecutionResult object
        user_message: Optional user message to detect patterns like "primeiras N linhas"
        
    Returns:
        Formatted string with actual outputs from operations
    """
    import json
    import re
    
    # Detect if user asked for first N lines
    max_lines = None
    if user_message:
        # Patterns: "primeiras 10 linhas", "first 10 lines", "mostre as 10 primeiras", etc.
        patterns = [
            r'(?:primeiras?|first|mostre as|show the first)\s+(\d+)\s+(?:linhas?|lines)',
            r'(\d+)\s+(?:primeiras?|first)\s+(?:linhas?|lines)',
        ]
        for pattern in patterns:
            match = re.search(pattern, user_message.lower())
            if match:
                max_lines = int(match.group(1))
                break
    
    if not execution_result.steps:
        if execution_result.success:
            return "✅ Completed successfully (no operations needed)."
        else:
            return "❌ Execution failed."
    
    output_lines = []
    
    # If user requested specific number of lines, check if we have a shell command that extracted them
    # In that case, skip showing the full file read and show only the extracted lines
    skip_file_read = False
    if max_lines is not None:
        # Check if there's a shell command that extracted lines (like head, tail)
        for step in execution_result.steps:
            if (step.tool == "shell" and step.operation == "execute_command" and 
                step.success and step.arguments.get('command', '').startswith(('head', 'tail'))):
                skip_file_read = True
                break
    
    for step in execution_result.steps:
        if step.success:
            # Format successful step output
            if step.output is not None:
                # Special formatting for list_directory
                if step.tool == "filesystem" and step.operation == "list_directory":
                    if isinstance(step.output, dict) and "entries" in step.output:
                        entries = step.output["entries"]
                        path = step.output.get("path", "unknown")
                        output_lines.append(f"📁 Directory: {path}\n")
                        
                        if not entries:
                            output_lines.append("(empty directory)")
                        else:
                            for entry in entries:
                                icon = "📁" if entry.get("is_directory") else "📄"
                                name = entry.get("name", "unknown")
                                size_info = ""
                                if entry.get("is_file") and "size" in entry:
                                    size = entry["size"]
                                    if size < 1024:
                                        size_info = f" ({size} B)"
                                    elif size < 1024 * 1024:
                                        size_info = f" ({size / 1024:.1f} KB)"
                                    else:
                                        size_info = f" ({size / (1024 * 1024):.1f} MB)"
                                output_lines.append(f"  {icon} {name}{size_info}")
                    else:
                        # Fallback for other formats
                        try:
                            formatted = json.dumps(step.output, indent=2, ensure_ascii=False)
                            output_lines.append(f"{step.tool}.{step.operation}:\n{formatted}")
                        except (TypeError, ValueError):
                            output_lines.append(f"{step.tool}.{step.operation}:\n{step.output}")
                # Special formatting for read_file
                elif step.tool == "filesystem" and step.operation == "read_file":
                    # Skip showing full file if we have a shell command that extracted lines
                    if skip_file_read:
                        continue  # Skip this step, the shell command will show the extracted lines
                    
                    if isinstance(step.output, dict) and "content" in step.output:
                        content = step.output["content"]
                        path = step.output.get("path", "unknown")
                        size = step.output.get("size", 0)
                        lines = step.output.get("lines", 0)
                        
                        # Format file size
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f} KB"
                        else:
                            size_str = f"{size / (1024 * 1024):.1f} MB"
                        
                        # Show content with smart limiting
                        content_lines = content.splitlines()
                        total_lines = len(content_lines)
                        
                        # Use max_lines from user request if available, otherwise default to 1000
                        limit = max_lines if max_lines is not None else 1000
                        
                        # Only show file info if we're showing the full file or if it's a large file
                        if max_lines is None or total_lines <= limit:
                            output_lines.append(f"📄 File: {path}")
                            output_lines.append(f"Size: {size_str} | Lines: {lines}\n")
                        
                        if total_lines > limit:
                            output_lines.append("```")
                            output_lines.append("\n".join(content_lines[:limit]))
                            if max_lines is not None:
                                output_lines.append(f"\n... (showing first {limit} of {total_lines} lines as requested)")
                            else:
                                output_lines.append(f"\n... (truncated, showing first {limit} of {total_lines} lines)")
                            output_lines.append("```")
                        else:
                            output_lines.append("```")
                            output_lines.append(content)
                            output_lines.append("```")
                    else:
                        try:
                            formatted = json.dumps(step.output, indent=2, ensure_ascii=False)
                            output_lines.append(f"{step.tool}.{step.operation}:\n{formatted}")
                        except (TypeError, ValueError):
                            output_lines.append(f"{step.tool}.{step.operation}:\n{step.output}")
                # Special formatting for shell.execute_command (text processing)
                elif step.tool == "shell" and step.operation == "execute_command":
                    command = step.arguments.get('command', 'unknown')
                    
                    # Handle different output formats
                    if isinstance(step.output, dict):
                        # Shell tool returns dict with stdout, stderr, return_code
                        stdout = step.output.get('stdout', '')
                        stderr = step.output.get('stderr', '')
                        return_code = step.output.get('return_code', 0)
                        
                        if return_code == 0 and stdout:
                            # Show command and output
                            output_lines.append(f"💻 Command: {command}\n")
                            output_lines.append("```")
                            output_lines.append(stdout.strip())
                            output_lines.append("```")
                        elif stderr:
                            output_lines.append(f"💻 Command: {command}\n")
                            output_lines.append(f"❌ Error (exit code {return_code}):\n```\n{stderr.strip()}\n```")
                        else:
                            output_lines.append(f"💻 Command: {command}\n(no output)")
                    elif isinstance(step.output, str):
                        # Direct string output
                        output_lines.append(f"💻 Command: {command}\n")
                        if step.output.strip():
                            output_lines.append("```")
                            output_lines.append(step.output.strip())
                            output_lines.append("```")
                        else:
                            output_lines.append("(no output)")
                    else:
                        # Fallback for other formats
                        try:
                            formatted = json.dumps(step.output, indent=2, ensure_ascii=False)
                            output_lines.append(f"{step.tool}.{step.operation}:\n{formatted}")
                        except (TypeError, ValueError):
                            output_lines.append(f"{step.tool}.{step.operation}:\n{step.output}")
                # Default formatting for other operations
                else:
                    if isinstance(step.output, (dict, list)):
                        try:
                            formatted = json.dumps(step.output, indent=2, ensure_ascii=False)
                            output_lines.append(f"{step.tool}.{step.operation}:\n{formatted}")
                        except (TypeError, ValueError):
                            output_lines.append(f"{step.tool}.{step.operation}:\n{step.output}")
                    elif isinstance(step.output, str):
                        output_lines.append(f"{step.tool}.{step.operation}:\n{step.output}")
                    else:
                        output_lines.append(f"{step.tool}.{step.operation}:\n{str(step.output)}")
            else:
                output_lines.append(f"{step.tool}.{step.operation}: ✅ Completed")
        else:
            # Format failed step
            error_msg = step.error or "Unknown error"
            output_lines.append(f"{step.tool}.{step.operation}: ❌ Failed - {error_msg}")
    
    if not execution_result.success and execution_result.stopped_at_step:
        output_lines.append(f"\n⚠️ Execution stopped at step {execution_result.stopped_at_step}.")
    
    return "\n".join(output_lines) if output_lines else "No output available."


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    storage: Storage = Depends(get_storage),
) -> CreateSessionResponse:
    """Create a new chat session.

    Args:
        request: Create session request with optional title
        storage: Storage instance (injected)

    Returns:
        CreateSessionResponse with session_id and title
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    start_time = time.time()
    log_event(logger, "api.request.started", endpoint="/sessions", method="POST")

    try:
        session = await storage.create_session(request.title)
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions", method="POST", status="201").inc()
        api_request_duration_seconds.labels(endpoint="/sessions").observe(duration)

        log_event(
            logger,
            "api.request.completed",
            endpoint="/sessions",
            method="POST",
            status=201,
            duration_ms=duration * 1000,
            session_id=session.session_id,
        )

        return CreateSessionResponse(session_id=session.session_id, title=session.title)
    except StorageError as e:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions", method="POST", status="500").inc()
        api_request_duration_seconds.labels(endpoint="/sessions").observe(duration)

        log_event(
            logger,
            "api.request.failed",
            level="ERROR",
            endpoint="/sessions",
            method="POST",
            status=500,
            duration_ms=duration * 1000,
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to create session"},
        ) from e


@router.get("/sessions", response_model=SessionsListResponse, status_code=status.HTTP_200_OK)
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    storage: Storage = Depends(get_storage),
) -> SessionsListResponse:
    """List chat sessions with pagination.

    Args:
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip
        storage: Storage instance (injected)

    Returns:
        SessionsListResponse with list of session summaries
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    start_time = time.time()
    log_event(logger, "api.request.started", endpoint="/sessions", method="GET")

    try:
        summaries = await storage.list_sessions(limit, offset)
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions", method="GET", status="200").inc()
        api_request_duration_seconds.labels(endpoint="/sessions").observe(duration)

        return SessionsListResponse(
            sessions=[s.model_dump() for s in summaries],
            limit=limit,
            offset=offset,
        )
    except StorageError as e:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions", method="GET", status="500").inc()
        api_request_duration_seconds.labels(endpoint="/sessions").observe(duration)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to list sessions"},
        ) from e


@router.delete("/sessions/{session_id}", status_code=status.HTTP_200_OK)
async def delete_session(
    session_id: str,
    storage: Storage = Depends(get_storage),
) -> Dict[str, str]:
    """Delete a session and all its messages.

    Args:
        session_id: Session identifier
        storage: Storage instance (injected)

    Returns:
        Success message
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    set_run_id(session_id)

    start_time = time.time()
    log_event(logger, "api.request.started", endpoint=f"/sessions/{session_id}", method="DELETE")

    try:
        await storage.delete_session(session_id)
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions/{session_id}", method="DELETE", status="200").inc()
        api_request_duration_seconds.labels(endpoint="/sessions/{session_id}").observe(duration)

        log_event(
            logger,
            "api.request.completed",
            endpoint=f"/sessions/{session_id}",
            method="DELETE",
            status=200,
            duration_ms=duration * 1000,
            session_id=session_id,
        )

        return {"status": "deleted", "session_id": session_id}
    except NotFoundError as e:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions/{session_id}", method="DELETE", status="404").inc()
        api_request_duration_seconds.labels(endpoint="/sessions/{session_id}").observe(duration)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Session not found: {session_id}"},
        ) from e
    except StorageError as e:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions/{session_id}", method="DELETE", status="500").inc()
        api_request_duration_seconds.labels(endpoint="/sessions/{session_id}").observe(duration)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to delete session"},
        ) from e


@router.get("/sessions/{session_id}", response_model=SessionResponse, status_code=status.HTTP_200_OK)
async def get_session(
    session_id: str,
    storage: Storage = Depends(get_storage),
) -> SessionResponse:
    """Get a session by ID with all messages.

    Args:
        session_id: Session identifier
        storage: Storage instance (injected)

    Returns:
        SessionResponse with full session data
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    set_run_id(session_id)  # Use session_id as run_id for observability

    start_time = time.time()
    log_event(logger, "api.request.started", endpoint=f"/sessions/{session_id}", method="GET")

    try:
        session = await storage.get_session(session_id)
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions/{session_id}", method="GET", status="200").inc()
        api_request_duration_seconds.labels(endpoint="/sessions/{session_id}").observe(duration)

        return SessionResponse(
            session_id=session.session_id,
            title=session.title,
            messages=[
                MessageResponse(
                    message_id=m.message_id,
                    role=m.role.value,
                    content=m.content,
                    created_at=m.created_at,
                    plan_result=m.plan_result,
                    execution_result=m.execution_result,
                )
                for m in session.messages
            ],
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
    except NotFoundError as e:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions/{session_id}", method="GET", status="404").inc()
        api_request_duration_seconds.labels(endpoint="/sessions/{session_id}").observe(duration)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Session not found: {session_id}"},
        ) from e
    except StorageError as e:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/sessions/{session_id}", method="GET", status="500").inc()
        api_request_duration_seconds.labels(endpoint="/sessions/{session_id}").observe(duration)

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to get session"},
        ) from e


@router.post(
    "/sessions/{session_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    session_id: str,
    request: MessageRequest,
    planner: Planner = Depends(get_planner),
    storage: Storage = Depends(get_storage),
    llm_provider = Depends(get_llm_provider),
) -> MessageResponse:
    """Send a message to a session and get AI response.

    This endpoint:
    1. Saves the user message
    2. Uses Planner to generate a plan from the message + context
    3. Optionally executes the plan
    4. Saves the assistant response
    5. Returns the assistant message

    Args:
        session_id: Session identifier
        request: Message request with content
        planner: Planner instance (injected)
        storage: Storage instance (injected)

    Returns:
        MessageResponse with assistant's response
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    set_run_id(session_id)

    start_time = time.time()
    log_event(
        logger,
        "api.request.started",
        endpoint=f"/sessions/{session_id}/messages",
        method="POST",
        session_id=session_id,
    )

    try:
        # Step 1: Get session to build context
        session = await storage.get_session(session_id)

        # Step 2: Save user message
        user_message = await storage.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.content,
        )

        # Step 3: Build context from previous messages
        # Use last few messages as context for the planner
        context_messages = session.messages[-5:] if len(session.messages) > 5 else session.messages
        context = {
            "previous_messages": [
                {"role": m.role.value, "content": m.content} for m in context_messages
            ]
        }

        # Step 4: Generate plan using Planner
        plan_result = None
        execution_result = None
        assistant_content = ""
        hitl_enabled = False
        steps_requiring_approval = []

        try:
            with trace_span("plan", attributes={"session_id": session_id}):
                plan_result_obj = await planner.plan(request.content, context)

            plan_result = {
                "plan": plan_result_obj.plan.model_dump(),
                "diagnostics": plan_result_obj.diagnostics.model_dump(),
            }

            logger.info(
                f"[{request_id}] Planning succeeded - plan_id: {plan_result_obj.plan.plan_id}, "
                f"steps: {len(plan_result_obj.plan.steps)}"
            )

            # Step 5: Check if plan is empty (no tools needed)
            config = get_config()
            if not plan_result_obj.plan.steps:
                # Empty plan means no tools are needed - answer directly
                # Use LLM to generate response based on user's question
                assistant_content = await _generate_final_response(
                    llm_provider, request.content, None, config
                )
            else:
                # Step 6: Check if HITL is enabled and if plan has operations requiring approval
                hitl_enabled = config.human_in_the_loop.enabled
                steps_requiring_approval = get_operations_requiring_approval(plan_result_obj.plan)

                if hitl_enabled and steps_requiring_approval:
                    # HITL enabled and plan has operations requiring approval
                    # Execute only safe operations (read-only), skip operations requiring approval
                    safe_steps = [
                        step
                        for step in plan_result_obj.plan.steps
                        if step not in steps_requiring_approval
                    ]

                    if safe_steps:
                        # Create a plan with only safe steps and execute
                        from agent.runtime.schema import Plan

                        safe_plan = Plan(
                            plan_id=f"{plan_result_obj.plan.plan_id}-safe",
                            objective=plan_result_obj.plan.objective,
                            steps=safe_steps,
                            estimated_time_seconds=plan_result_obj.plan.estimated_time_seconds,
                            notes=plan_result_obj.plan.notes,
                        )

                        execution_policy = None
                        if request.execution_policy:
                            execution_policy = ExecutionPolicy(**request.execution_policy)

                        tool_registry = get_tool_registry(config)
                        executor = Executor(config, tool_registry, execution_policy)

                        execution_result_obj = await executor.execute(safe_plan)
                        execution_result = execution_result_obj.model_dump()

                        logger.info(
                            f"[{request_id}] Safe operations executed - success: {execution_result_obj.success}"
                        )

                        # Generate final response from LLM
                        from agent.runtime.schema import ExecutionResult
                        execution_result_for_llm = ExecutionResult(**execution_result)
                        assistant_content = await _generate_final_response(
                            llm_provider, request.content, execution_result_for_llm, config
                        )
                        
                        if steps_requiring_approval:
                            assistant_content += f"\n\n⚠️  The following operations require approval:\n"
                            for step in steps_requiring_approval:
                                assistant_content += f"  • Step {step.step_id}: {step.tool.value}.{step.operation} - {step.rationale}\n"
                            assistant_content += f"\nPlease review and approve/reject these operations."
                    else:
                        # All steps require approval - just show the plan
                        assistant_content = f"📋 Plan generated. All operations require approval.\n\n"
                        assistant_content += f"⚠️  The following {len(steps_requiring_approval)} operation(s) require approval:\n"
                        for step in steps_requiring_approval:
                            assistant_content += f"  • Step {step.step_id}: {step.tool.value}.{step.operation} - {step.rationale}\n"
                        assistant_content += f"\nPlease review and approve/reject these operations."
                elif hitl_enabled and not steps_requiring_approval:
                    # HITL enabled but no operations require approval - execute automatically
                    execution_policy = None
                    if request.execution_policy:
                        execution_policy = ExecutionPolicy(**request.execution_policy)

                    tool_registry = get_tool_registry(config)
                    executor = Executor(config, tool_registry, execution_policy)

                    execution_result_obj = await executor.execute(plan_result_obj.plan)
                    execution_result = execution_result_obj.model_dump()

                    logger.info(
                        f"[{request_id}] Execution completed (HITL enabled, no approval needed) - success: {execution_result_obj.success}"
                    )

                    # Generate final response from LLM
                    from agent.runtime.schema import ExecutionResult
                    execution_result_for_llm = ExecutionResult(**execution_result)
                    assistant_content = await _generate_final_response(
                        llm_provider, request.content, execution_result_for_llm, config
                    )
                else:
                    # HITL disabled - execute everything
                    execution_policy = None
                    if request.execution_policy:
                        execution_policy = ExecutionPolicy(**request.execution_policy)

                    tool_registry = get_tool_registry(config)
                    executor = Executor(config, tool_registry, execution_policy)

                    execution_result_obj = await executor.execute(plan_result_obj.plan)
                    execution_result = execution_result_obj.model_dump()

                    logger.info(
                        f"[{request_id}] Execution completed - success: {execution_result_obj.success}"
                    )

                    # Generate final response from LLM
                    from agent.runtime.schema import ExecutionResult
                    execution_result_for_llm = ExecutionResult(**execution_result)
                    assistant_content = await _generate_final_response(
                        llm_provider, request.content, execution_result_for_llm, config
                    )

        except InvalidPlanError as e:
            logger.error(f"[{request_id}] Invalid plan error: {e}")
            assistant_content = f"❌ Planning failed: {str(e)}\n"
            if hasattr(e, "diagnostics") and e.diagnostics:
                assistant_content += f"Diagnostics: {e.diagnostics.model_dump()}\n"
        except LLMCommunicationError as e:
            logger.error(f"[{request_id}] LLM communication error: {e}")
            assistant_content = f"❌ Communication error: {str(e)}\n"
        except PlanningError as e:
            logger.error(f"[{request_id}] Planning error: {e}")
            assistant_content = f"❌ Planning failed: {str(e)}\n"
        except Exception as e:
            logger.exception(f"[{request_id}] Unexpected error: {e}")
            assistant_content = f"❌ Unexpected error: {str(e)}\n"

        # Step 7: Save assistant message
        assistant_message = await storage.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=assistant_content,
            plan_result=plan_result,
            execution_result=execution_result,
        )

        # Step 8: Update session title if it's the first message
        if len(session.messages) == 0:
            # Use first 50 chars of first user message as title
            title = request.content[:50] + ("..." if len(request.content) > 50 else "")
            await storage.update_session_title(session_id, title)

        duration = time.time() - start_time
        api_requests_total.labels(
            endpoint=f"/sessions/{session_id}/messages", method="POST", status="201"
        ).inc()
        api_request_duration_seconds.labels(endpoint=f"/sessions/{session_id}/messages").observe(
            duration
        )

        log_event(
            logger,
            "api.request.completed",
            endpoint=f"/sessions/{session_id}/messages",
            method="POST",
            status=201,
            duration_ms=duration * 1000,
            session_id=session_id,
        )

        # Include pending approval steps if any
        pending_steps = None
        if hitl_enabled and steps_requiring_approval:
            pending_steps = [step.step_id for step in steps_requiring_approval]

        return MessageResponse(
            message_id=assistant_message.message_id,
            role=assistant_message.role.value,
            content=assistant_message.content,
            created_at=assistant_message.created_at,
            plan_result=assistant_message.plan_result,
            execution_result=assistant_message.execution_result,
            pending_approval_steps=pending_steps,
        )

    except NotFoundError as e:
        duration = time.time() - start_time
        api_requests_total.labels(
            endpoint=f"/sessions/{session_id}/messages", method="POST", status="404"
        ).inc()
        api_request_duration_seconds.labels(endpoint=f"/sessions/{session_id}/messages").observe(
            duration
        )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Session not found: {session_id}"},
        ) from e
    except StorageError as e:
        duration = time.time() - start_time
        api_requests_total.labels(
            endpoint=f"/sessions/{session_id}/messages", method="POST", status="500"
        ).inc()
        api_request_duration_seconds.labels(endpoint=f"/sessions/{session_id}/messages").observe(
            duration
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to send message"},
        ) from e
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e


@router.post(
    "/sessions/{session_id}/messages/{message_id}/approve",
    response_model=ApproveOperationsResponse,
    status_code=status.HTTP_200_OK,
)
async def approve_operations(
    session_id: str,
    message_id: str,
    request: ApproveOperationsRequest,
    storage: Storage = Depends(get_storage),
    llm_provider = Depends(get_llm_provider),
) -> ApproveOperationsResponse:
    """Approve operations that require approval.

    Args:
        session_id: Session identifier
        message_id: Message identifier containing the plan
        request: Approval request with step IDs to approve
        storage: Storage instance (injected)

    Returns:
        ApproveOperationsResponse with execution result
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    set_run_id(session_id)

    start_time = time.time()
    log_event(
        logger,
        "api.request.started",
        endpoint=f"/sessions/{session_id}/messages/{message_id}/approve",
        method="POST",
        session_id=session_id,
        message_id=message_id,
    )

    try:
        # Get session and message
        session = await storage.get_session(session_id)
        message = None
        for msg in session.messages:
            if msg.message_id == message_id:
                message = msg
                break

        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Message not found: {message_id}"},
            )

        if not message.plan_result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Message does not contain a plan"},
            )

        # Parse plan
        from agent.runtime.schema import Plan

        plan_dict = message.plan_result.get("plan", {})
        plan = Plan(**plan_dict)

        # Filter steps to approve
        steps_to_approve = [step for step in plan.steps if step.step_id in request.step_ids]

        if not steps_to_approve:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "No valid steps to approve"},
            )

        # Create plan with only approved steps
        approved_plan = Plan(
            plan_id=f"{plan.plan_id}-approved",
            objective=plan.objective,
            steps=steps_to_approve,
            estimated_time_seconds=plan.estimated_time_seconds,
            notes=plan.notes,
        )

        # Execute approved steps
        config = get_config()
        tool_registry = get_tool_registry(config)
        executor = Executor(config, tool_registry, None)

        execution_result_obj = await executor.execute(approved_plan)
        execution_result = execution_result_obj.model_dump()

        # Generate final response from LLM
        from agent.runtime.schema import ExecutionResult
        execution_result_obj = ExecutionResult(**execution_result)
        config = get_config()
        
        # Get original user message from the plan message (look for previous user message)
        user_message = "Approved operations executed"
        for prev_msg in session.messages:
            if prev_msg.role == MessageRole.USER and prev_msg.message_id != message_id:
                user_message = prev_msg.content
                break
        
        output_content = await _generate_final_response(
            llm_provider, user_message, execution_result_obj, config
        )
        
        # Save execution result to message
        await storage.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=output_content,
            plan_result=None,
            execution_result=execution_result,
        )

        duration = time.time() - start_time
        api_requests_total.labels(
            endpoint=f"/sessions/{session_id}/messages/{message_id}/approve",
            method="POST",
            status="200",
        ).inc()
        api_request_duration_seconds.labels(
            endpoint=f"/sessions/{session_id}/messages/{message_id}/approve"
        ).observe(duration)

        return ApproveOperationsResponse(
            message_id=message_id,
            execution_result=execution_result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e


@router.post(
    "/sessions/{session_id}/messages/{message_id}/reject",
    status_code=status.HTTP_200_OK,
)
async def reject_operations(
    session_id: str,
    message_id: str,
    request: RejectOperationsRequest,
    storage: Storage = Depends(get_storage),
) -> Dict[str, str]:
    """Reject operations that require approval.

    Args:
        session_id: Session identifier
        message_id: Message identifier containing the plan
        request: Rejection request with step IDs to reject
        storage: Storage instance (injected)

    Returns:
        Success message
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    set_run_id(session_id)

    start_time = time.time()
    log_event(
        logger,
        "api.request.started",
        endpoint=f"/sessions/{session_id}/messages/{message_id}/reject",
        method="POST",
        session_id=session_id,
        message_id=message_id,
    )

    try:
        # Save rejection message
        reason = request.reason or "Operations rejected by user"
        await storage.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=f"❌ Rejected {len(request.step_ids)} operation(s). Reason: {reason}",
            plan_result=None,
            execution_result=None,
        )

        duration = time.time() - start_time
        api_requests_total.labels(
            endpoint=f"/sessions/{session_id}/messages/{message_id}/reject",
            method="POST",
            status="200",
        ).inc()
        api_request_duration_seconds.labels(
            endpoint=f"/sessions/{session_id}/messages/{message_id}/reject"
        ).observe(duration)

        return {"status": "rejected", "message_id": message_id}

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e

"""Session/chat API routes."""

import time
from typing import Dict
from agent.id import ascending

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from agent.observability import (
    api_request_duration_seconds,
    api_requests_total,
    get_logger,
    log_event,
    set_request_id,
    trace_span,
)
from agent.runtime.bus import EventType, publish
from agent.runtime.event_helpers import publish_if_session_exists
from agent.storage import MessageRole, NotFoundError, Storage, StorageError
from api.dependencies import get_config, get_storage, get_tool_registry
from api.schemas.session import (
    CreateSessionRequest,
    CreateSessionResponse,
    MessageRequest,
    MessageResponse,
    SessionResponse,
    SessionsListResponse,
)

router = APIRouter()
logger = get_logger("api.session", "api")


from agent.runtime.event_helpers import session_exists


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
    # Request ID for correlation (not persisted, just for logs/events)
    # OpenCode doesn't have a specific prefix for request IDs, so we use a simple UUID-like approach
    # But we'll use message prefix for consistency with ID system
    from agent.id import ascending
    request_id = ascending("message")
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
    # Request ID for correlation (not persisted, just for logs/events)
    # OpenCode doesn't have a specific prefix for request IDs, so we use a simple UUID-like approach
    # But we'll use message prefix for consistency with ID system
    from agent.id import ascending
    request_id = ascending("message")
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
    # Request ID for correlation (not persisted, just for logs/events)
    # OpenCode doesn't have a specific prefix for request IDs, so we use a simple UUID-like approach
    # But we'll use message prefix for consistency with ID system
    from agent.id import ascending
    request_id = ascending("message")
    set_request_id(request_id)

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
    # Request ID for correlation (not persisted, just for logs/events)
    # OpenCode doesn't have a specific prefix for request IDs, so we use a simple UUID-like approach
    # But we'll use message prefix for consistency with ID system
    from agent.id import ascending
    request_id = ascending("message")
    set_request_id(request_id)

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
        
        # Log at debug level instead of error - 404 is expected when session was deleted
        logger.debug(f"Session not found (expected if deleted): {session_id}")

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


async def _process_message_background(
    session_id: str,
    request: MessageRequest,
    storage: Storage,
):
    """Process message in background using LangChain executor.
    
    This function runs after send_message returns immediately.
    Publishes all events via Bus for WebSocket streaming.
    
    LangChainExecutor creates the LLM internally from config, so no llm_provider needed.
    """
    # Request ID for correlation (not persisted, just for logs/events)
    # OpenCode doesn't have a specific prefix for request IDs, so we use a simple UUID-like approach
    # But we'll use message prefix for consistency with ID system
    from agent.id import ascending
    request_id = ascending("message")
    set_request_id(request_id)
    
    start_time = time.time()
    
    # Log that background processing started
    log_event(
        logger,
        "message.processing.started",
        request_id=request_id,
        session_id=session_id,
        message_content=request.content[:100],
    )
    
    try:
        # Step 1: Get session to build context
        session = await storage.get_session(session_id)

        # Step 2: Build conversation history from previous messages
        conversation_history = []
        for msg in session.messages[-10:]:  # Last 10 messages for context
            conversation_history.append({
                "role": msg.role.value,
                "content": msg.content
            })

        # Step 3: Use LangChainExecutor - LLM uses tools via LangChain agents
        config = get_config()
        tool_registry = get_tool_registry()
        
        from agent.runtime.langchain_executor import get_shared_executor
        
        # Publish execution started event (only if session still exists)
        await publish_if_session_exists(
            EventType.EXECUTION_STARTED,
            {"session_id": session_id},
            storage,
        )
        
        # Get shared executor instance (singleton) - reduces resource usage
        # The executor is shared across all sessions, with session_id passed to run()
        executor = await get_shared_executor(
            config=config,
            tool_registry=tool_registry,
            storage=storage,  # Pass storage to enable session existence checks
        )
        
        # Run LangChain agent execution with session_id
        result = await executor.run(
            user_message=request.content,
            conversation_history=conversation_history if conversation_history else None,
            session_id=session_id,  # Pass session_id to run() method
            storage=storage,
        )
        
        # Extract results
        assistant_content = result.get("response", "")
        
        # Log if response is empty
        if not assistant_content:
            log_event(
                logger,
                "message.processing.empty_response",
                request_id=request_id,
                session_id=session_id,
                result_keys=list(result.keys()),
                level="WARNING",
            )
        
        # Check if session still exists before continuing
        if not await session_exists(storage, session_id):
            logger.warning(f"[{request_id}] Session {session_id} was deleted during processing, stopping")
            return
        
        # Publish execution completed event (only if session still exists)
        await publish_if_session_exists(
            EventType.EXECUTION_COMPLETED,
            {
                "session_id": session_id,
                "success": result.get("success", False),
                "iterations": result.get("iterations", 0),
            },
            storage,
        )
        
        # Step 4: Save assistant message (even if empty, to maintain conversation flow)
        # Check again before saving (session might have been deleted between checks)
        if not await session_exists(storage, session_id):
            logger.warning(f"[{request_id}] Session {session_id} was deleted before saving message, stopping")
            return
        
        assistant_message = await storage.add_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=assistant_content if assistant_content else "(No response generated)",
        )
        
        # Publish event: assistant message added (only if session still exists)
        await publish_if_session_exists(
            EventType.SESSION_MESSAGE_ADDED,
            {
                "session_id": session_id,
                "message_id": assistant_message.message_id,
                "role": "assistant",
            },
            storage,
        )
        
        # Publish session updated event (only if session still exists)
        await publish_if_session_exists(
            EventType.SESSION_UPDATED,
            {
                "session_id": session_id,
                "message_count": len(session.messages) + 2,  # +2 for user and assistant
            },
            storage,
        )

        # Step 5: Update session title if it's the first message
        if len(session.messages) == 0:
            # Use first 50 chars of first user message as title
            title = request.content[:50] + ("..." if len(request.content) > 50 else "")
            await storage.update_session_title(session_id, title)

        duration = time.time() - start_time
        log_event(
            logger,
            "message.processing.completed",
            request_id=request_id,
            session_id=session_id,
            duration_ms=duration * 1000,
        )
    except NotFoundError as e:
        logger.error(f"[{request_id}] Session not found: {session_id}")
        await publish(EventType.EXECUTION_FAILED, {
            "session_id": session_id,
            "error": f"Session not found: {session_id}",
        })
    except StorageError as e:
        logger.error(f"[{request_id}] Storage error: {e}")
        await publish(EventType.EXECUTION_FAILED, {
            "session_id": session_id,
            "error": f"Storage error: {str(e)}",
        })
    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error in background processing: {e}")
        await publish(EventType.EXECUTION_FAILED, {
            "session_id": session_id,
            "error": str(e),
        })
        # Re-raise to see in logs
        raise


@router.post(
    "/sessions/{session_id}/messages",
    response_model=MessageResponse,
    status_code=status.HTTP_202_ACCEPTED,  # Accepted, processing in background
)
async def send_message(
    session_id: str,
    request: MessageRequest,
    background_tasks: BackgroundTasks,
    storage: Storage = Depends(get_storage),
) -> MessageResponse:
    """Send a message to a session (non-blocking, like OpenCode).

    This endpoint:
    1. Saves the user message immediately
    2. Returns immediately (202 Accepted)
    3. Processes plan + execution in background
    4. Publishes events via Bus (SSE) for real-time updates

    Args:
        session_id: Session identifier
        request: Message request with content
        planner: Planner instance (injected)
        storage: Storage instance (injected)

    Returns:
        MessageResponse with user message (assistant response comes via SSE events)
    """
    # Request ID for correlation (not persisted, just for logs/events)
    # OpenCode doesn't have a specific prefix for request IDs, so we use a simple UUID-like approach
    # But we'll use message prefix for consistency with ID system
    from agent.id import ascending
    request_id = ascending("message")
    set_request_id(request_id)

    start_time = time.time()
    log_event(
        logger,
        "api.request.started",
        endpoint=f"/sessions/{session_id}/messages",
        method="POST",
        session_id=session_id,
    )

    try:
        # Step 1: Get session
        session = await storage.get_session(session_id)

        # Step 2: Save user message immediately
        user_message = await storage.add_message(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.content,
        )
        
        # Publish event: message added (only if session still exists)
        await publish_if_session_exists(
            EventType.SESSION_MESSAGE_ADDED,
            {
                "session_id": session_id,
                "message_id": user_message.message_id,
                "role": "user",
            },
            storage,
        )
        
        # Update session title if it's the first message
        if len(session.messages) == 0:
            title = request.content[:50] + ("..." if len(request.content) > 50 else "")
            await storage.update_session_title(session_id, title)

        # Step 3: Process in background (non-blocking)
        # LangChainExecutor loads all LLM providers and tools through LangChain
        # No need to pass llm_provider - it's created internally from config
        background_tasks.add_task(
            _process_message_background,
            session_id=session_id,
            request=request,
            storage=storage,
        )
        
        # Log task creation
        log_event(
            logger,
            "message.processing.background_task_created",
            session_id=session_id,
        )

        duration = time.time() - start_time
        api_requests_total.labels(
            endpoint=f"/sessions/{session_id}/messages", method="POST", status="202"
        ).inc()
        api_request_duration_seconds.labels(endpoint=f"/sessions/{session_id}/messages").observe(
            duration
        )

        log_event(
            logger,
            "api.request.completed",
            endpoint=f"/sessions/{session_id}/messages",
            method="POST",
            status=202,
            duration_ms=duration * 1000,
            session_id=session_id,
        )

        # Return immediately with user message
        return MessageResponse(
            message_id=user_message.message_id,
            role=user_message.role.value,
            content=user_message.content,
            created_at=user_message.created_at,
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


# Removed approve/reject endpoints - no longer needed with direct tool calling

"""SSE endpoint for real-time event streaming.

Uses Server-Sent Events (SSE) for streaming agent reasoning and execution updates.
SSE is simpler and more reliable for one-way event streaming from server to client.
"""

import asyncio
import json
import time
from typing import Set

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from agent.observability import get_logger
from agent.runtime.bus import Event, EventType, get_bus
from agent.runtime.event_helpers import session_exists
from api.dependencies import get_storage

router = APIRouter()
logger = get_logger("api.events", "api")

# Store active WebSocket connections
_active_websocket_connections: Set[WebSocket] = set()

# Store active SSE connections (for backward compatibility)
_active_sse_connections: Set[asyncio.Queue] = set()


def _format_event_data(event: Event) -> dict:
    """Format event as JSON-serializable dict.
    
    Args:
        event: Event to format
        
    Returns:
        Dictionary with event data
    """
    return {
        "type": event.type.value,
        "properties": event.properties,
        "timestamp": event.timestamp,
    }


async def _event_stream(session_id: str):
    """Stream events via SSE.
    
    Args:
        session_id: Session ID to filter events. Only events for this session will be sent.
    
    Yields:
        SSE-formatted event strings
    """
    # Create queue for this connection
    queue = asyncio.Queue()
    _active_sse_connections.add(queue)
    
    # Get storage for session existence checks
    storage = get_storage()
    
    # Verify session exists before starting stream
    if not await session_exists(storage, session_id):
        logger.warning(f"SSE connection rejected - session does not exist: {session_id}")
        error_event = Event(
            type=EventType.SESSION_UPDATED,
            properties={"type": "server.error", "error": "Session not found", "session_id": session_id},
            timestamp=time.time(),
        )
        data = json.dumps(_format_event_data(error_event))
        yield f"data: {data}\n\n"
        return
    
    logger.info(f"SSE connection established (session_id={session_id})")
    
    # Send initial connection event
    initial_event = Event(
        type=EventType.SESSION_UPDATED,
        properties={"type": "server.connected", "session_id": session_id},
        timestamp=time.time(),
    )
    data = json.dumps(_format_event_data(initial_event))
    yield f"data: {data}\n\n"
    
    # Subscribe to all events via subscribe_all_async
    bus = get_bus()
    
    async def event_handler(event: Event):
        """Handle event from bus - filter by session_id (required)."""
        event_session_id = event.properties.get("session_id")
        
        # If event has no session_id, skip it (unless it's a system event)
        if not event_session_id:
            # Allow system events (heartbeat, connection, etc.) only if they match our session
            event_type = event.properties.get("type")
            if event_type in ("server.connected", "server.heartbeat"):
                # Check if the event's session_id matches (if present in properties)
                if event.properties.get("session_id") == session_id:
                    try:
                        await queue.put(event)
                    except Exception as e:
                        logger.error(f"Error queuing event: {e}")
            return
        
        # Only queue events for the specified session
        if event_session_id != session_id:
            return
        
        # Verify session still exists before queuing (prevents 404s)
        if not await session_exists(storage, event_session_id):
            logger.debug(f"Skipping event for deleted session: {event_session_id}")
            return
        
        # Queue the event
        try:
            await queue.put(event)
        except Exception as e:
            logger.error(f"Error queuing event: {e}")
    
    # Subscribe to all events
    bus.subscribe_all_async(event_handler)
    
    # Send heartbeat every 30s to prevent timeout
    last_heartbeat = time.time()
    heartbeat_interval = 30.0
    
    try:
        while True:
            try:
                # Wait for event with timeout for heartbeat
                timeout = max(0.1, heartbeat_interval - (time.time() - last_heartbeat))
                event = await asyncio.wait_for(queue.get(), timeout=timeout)
                data = json.dumps(_format_event_data(event))
                yield f"data: {data}\n\n"
            except asyncio.TimeoutError:
                # Send heartbeat
                if time.time() - last_heartbeat >= heartbeat_interval:
                    heartbeat_event = Event(
                        type=EventType.SESSION_UPDATED,
                        properties={"type": "server.heartbeat", "session_id": session_id},
                        timestamp=time.time(),
                    )
                    data = json.dumps(_format_event_data(heartbeat_event))
                    yield f"data: {data}\n\n"
                    last_heartbeat = time.time()
    except asyncio.CancelledError:
        logger.info("SSE connection cancelled")
    finally:
        # Cleanup
        _active_sse_connections.discard(queue)
        # Unsubscribe from all events
        try:
            bus.unsubscribe_all_async(event_handler)
        except Exception:
            pass
        logger.info("SSE connection closed")


@router.websocket("/events/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for subscribing to events.
    
    Streams all events from the Bus via WebSocket.
    Supports bidirectional communication and proper request_id propagation.
    
    Args:
        websocket: WebSocket connection
    """
    await websocket.accept()
    _active_websocket_connections.add(websocket)
    
    logger.info("WebSocket connection established")
    
    # Send initial connection event
    initial_event = Event(
        type=EventType.SESSION_UPDATED,
        properties={"type": "server.connected"},
        timestamp=time.time(),
    )
    await websocket.send_json(_format_event_data(initial_event))
    
    # Subscribe to all events via subscribe_all_async
    bus = get_bus()
    
    # Queue for events from bus
    event_queue = asyncio.Queue()
    
    async def event_handler(event: Event):
        """Handle event from bus."""
        try:
            await event_queue.put(event)
        except Exception as e:
            logger.error(f"Error queuing event: {e}")
    
    # Subscribe to all events
    bus.subscribe_all_async(event_handler)
    
    # Task to send heartbeat
    async def send_heartbeat():
        """Send heartbeat every 30s to keep connection alive."""
        while True:
            await asyncio.sleep(30.0)
            try:
                heartbeat_event = Event(
                    type=EventType.SESSION_UPDATED,
                    properties={"type": "server.heartbeat"},
                    timestamp=time.time(),
                )
                await websocket.send_json(_format_event_data(heartbeat_event))
            except Exception:
                break
    
    heartbeat_task = asyncio.create_task(send_heartbeat())
    
    try:
        # Task to receive events from bus and send via WebSocket
        async def send_events():
            """Send events from bus to WebSocket."""
            while True:
                try:
                    event = await event_queue.get()
                    await websocket.send_json(_format_event_data(event))
                except Exception as e:
                    logger.error(f"Error sending event via WebSocket: {e}")
                    break
        
        send_task = asyncio.create_task(send_events())
        
        # Task to receive messages from client (for ping/pong or future features)
        async def receive_messages():
            """Receive messages from client."""
            while True:
                try:
                    data = await websocket.receive_text()
                    # Handle ping/pong or other client messages
                    if data == "ping":
                        await websocket.send_text("pong")
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"Error receiving message from WebSocket: {e}")
                    break
        
        receive_task = asyncio.create_task(receive_messages())
        
        # Wait for any task to complete (connection closed)
        done, pending = await asyncio.wait(
            [send_task, receive_task, heartbeat_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cleanup
        _active_websocket_connections.discard(websocket)
        # Unsubscribe from all events
        try:
            bus.unsubscribe_all_async(event_handler)
        except Exception:
            pass
        # Cancel heartbeat task
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("WebSocket connection closed")


@router.get("/events/event")
async def event_endpoint_sse(session_id: str = Query(..., description="Session ID to filter events (required)")):
    """SSE endpoint for subscribing to events.
    
    Streams events from the Bus via Server-Sent Events.
    Only events for the specified session are sent.
    This prevents sending events from deleted or other sessions.
    
    Args:
        session_id: Session ID to filter events (required). Only events for this session will be sent.
    
    Returns:
        StreamingResponse with SSE events
    
    Raises:
        HTTPException: If session_id is not provided or session does not exist
    """
    logger.info(f"SSE connection request received (session_id={session_id})")
    
    return StreamingResponse(
        _event_stream(session_id=session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

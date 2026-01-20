"""Observability SSE endpoint for real-time system metrics."""

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from agent.observability.llm_metrics import get_llm_metrics
from agent.observability.system_metrics import get_metrics_collector

router = APIRouter()


async def _observability_stream():
    """Stream observability metrics via SSE.
    
    Yields:
        SSE-formatted event strings with metrics data
    """
    collector = get_metrics_collector()
    llm_metrics = get_llm_metrics()
    interval = 1.0  # Update interval in seconds
    
    # Send initial connection event
    initial_event = {
        "type": "connection",
        "data": {"status": "connected"},
    }
    yield f"data: {json.dumps(initial_event)}\n\n"
    
    # Send heartbeat every 30s to prevent timeout
    last_heartbeat = asyncio.get_event_loop().time()
    heartbeat_interval = 30.0
    
    try:
        while True:
            try:
                # Collect system metrics (global)
                system_metrics = collector.collect_all()
                
                # Collect LLM metrics (global and per-session)
                llm_global = llm_metrics.get_global_metrics()
                llm_per_session = llm_metrics.get_all_sessions_metrics()
                
                # Combine all metrics
                metrics = {
                    "type": "metrics",
                    "data": {
                        "timestamp": system_metrics.get("timestamp"),
                        "global": {
                            "system": system_metrics,
                            "llm": llm_global,
                        },
                        "sessions": {
                            session_id: {
                                "llm": session_metrics
                            }
                            for session_id, session_metrics in llm_per_session.items()
                        },
                    },
                }
                
                # Send metrics as SSE event
                yield f"data: {json.dumps(metrics)}\n\n"
                
                # Check if heartbeat is needed
                current_time = asyncio.get_event_loop().time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    heartbeat_event = {
                        "type": "heartbeat",
                        "data": {"timestamp": current_time},
                    }
                    yield f"data: {json.dumps(heartbeat_event)}\n\n"
                    last_heartbeat = current_time
                
                # Wait for next interval
                await asyncio.sleep(interval)
            except Exception as e:
                # Log error but continue streaming
                error_event = {
                    "type": "error",
                    "data": {"message": str(e)},
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass


@router.get("/observability/metrics")
async def observability_sse():
    """SSE endpoint for real-time system metrics streaming.
    
    Clients connect to this endpoint to receive real-time system metrics
    including CPU, memory, disk, network, and GPU (if available).
    
    The server streams metrics at regular intervals (default: 1 second).
    
    Returns:
        StreamingResponse with SSE events
    """
    return StreamingResponse(
        _observability_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )

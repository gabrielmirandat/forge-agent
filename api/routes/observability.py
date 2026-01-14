"""Observability WebSocket endpoint for real-time system metrics."""

import asyncio
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from agent.observability.llm_metrics import get_llm_metrics
from agent.observability.system_metrics import get_metrics_collector

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections for observability streaming."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task = None

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection to register
        """
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        """Unregister a WebSocket connection.
        
        Args:
            websocket: WebSocket connection to unregister
        """
        self.active_connections.discard(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific WebSocket connection.
        
        Args:
            message: Message to send (will be JSON-encoded)
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception:
            # Connection may be closed, remove it
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all active connections.
        
        Args:
            message: Message to broadcast (will be JSON-encoded)
        """
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def start_broadcasting(self, interval: float = 1.0):
        """Start broadcasting metrics at regular intervals.
        
        Args:
            interval: Interval between broadcasts in seconds (default: 1.0)
        """
        collector = get_metrics_collector()
        llm_metrics = get_llm_metrics()
        
        while True:
            try:
                # Collect system metrics (global)
                system_metrics = collector.collect_all()
                
                # Collect LLM metrics (global and per-session)
                llm_global = llm_metrics.get_global_metrics()
                llm_per_session = llm_metrics.get_all_sessions_metrics()
                
                # Combine all metrics
                metrics = {
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
                }
                
                # Broadcast to all connected clients
                if self.active_connections:
                    await self.broadcast(metrics)
                
                # Wait for next interval
                await asyncio.sleep(interval)
            except Exception as e:
                # Log error but continue broadcasting
                print(f"Error broadcasting metrics: {e}")
                await asyncio.sleep(interval)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/observability")
async def websocket_observability(websocket: WebSocket):
    """WebSocket endpoint for real-time system metrics streaming.
    
    Clients connect to this endpoint to receive real-time system metrics
    including CPU, memory, disk, network, and GPU (if available).
    
    The server broadcasts metrics at regular intervals (default: 1 second).
    """
    await manager.connect(websocket)
    
    try:
        # Start broadcasting if not already started
        if manager._broadcast_task is None or manager._broadcast_task.done():
            manager._broadcast_task = asyncio.create_task(
                manager.start_broadcasting(interval=1.0)
            )
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for any message from client (ping/pong or close)
                data = await websocket.receive_text()
                
                # Echo back or handle client messages
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

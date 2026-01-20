"""Event bus system for decoupled communication.

Similar to OpenCode's Bus system for publishing and subscribing to events.
"""

from typing import Any, Callable, Coroutine, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict

from agent.observability import get_logger, get_request_id


class EventType(str, Enum):
    """Event types."""

    # File events
    FILE_EDITED = "file.edited"
    FILE_READ = "file.read"
    FILE_DELETED = "file.deleted"
    FILE_CREATED = "file.created"
    
    # Session events
    SESSION_CREATED = "session.created"
    SESSION_MESSAGE_ADDED = "session.message.added"
    SESSION_UPDATED = "session.updated"
    
    # Execution events
    EXECUTION_STARTED = "execution.started"
    EXECUTION_STEP_STARTED = "execution.step.started"
    EXECUTION_STEP_COMPLETED = "execution.step.completed"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    
    # Tool events
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    
    # LLM thinking/progress events
    PLANNER_THINKING = "planner.thinking"  # Raw LLM response during planning
    EXECUTOR_PROGRESS = "executor.progress"  # Progress updates during execution
    LLM_RESPONSE = "llm.response"  # LLM text response (when no tool calls)
    LLM_REASONING = "llm.reasoning"  # LLM reasoning/thinking process
    TOOL_DECISION = "tool.decision"  # Model decision to use or not use tools


@dataclass
class Event:
    """Event data structure."""

    type: EventType
    properties: Dict[str, Any]
    timestamp: float


class Bus:
    """Event bus for publishing and subscribing to events."""

    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = defaultdict(list)
        self._async_subscribers: Dict[EventType, List[Callable[[Event], Coroutine[Any, Any, None]]]] = defaultdict(list)
        # All events subscriber (for SSE)
        self._all_subscribers: List[Callable[[Event], Coroutine[Any, Any, None]]] = []
        self.logger = get_logger("bus", "bus")
        self._lock = asyncio.Lock()

    async def publish(self, event_type: EventType, properties: Dict[str, Any]):
        """Publish an event.

        Args:
            event_type: Type of event
            properties: Event properties
        """
        import time

        # Include request_id from context if available
        enriched_properties = properties.copy()
        request_id = get_request_id()
        if request_id:
            enriched_properties["request_id"] = request_id

        event = Event(
            type=event_type,
            properties=enriched_properties,
            timestamp=time.time(),
        )

        self.logger.debug(f"Publishing event: {event_type}", properties=properties)

        # Notify sync subscribers
        for subscriber in self._subscribers.get(event_type, []):
            try:
                subscriber(event)
            except Exception as e:
                self.logger.error(f"Error in sync subscriber for {event_type}: {e}", exc_info=True)

        # Notify async subscribers
        async_subscribers = self._async_subscribers.get(event_type, [])
        all_subscribers = self._all_subscribers.copy()  # Copy to avoid modification during iteration
        
        if async_subscribers or all_subscribers:
            async with self._lock:
                tasks = []
                for subscriber in async_subscribers:
                    try:
                        tasks.append(asyncio.create_task(subscriber(event)))
                    except Exception as e:
                        self.logger.error(f"Error creating async subscriber task for {event_type}: {e}", exc_info=True)
                
                # Notify all-event subscribers (for SSE)
                for subscriber in all_subscribers:
                    try:
                        tasks.append(asyncio.create_task(subscriber(event)))
                    except Exception as e:
                        self.logger.error(f"Error creating all-event subscriber task: {e}", exc_info=True)

                # Wait for all subscribers to complete
                if tasks:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            self.logger.error(
                                f"Error in async subscriber {i} for {event_type}: {result}",
                                exc_info=True,
                            )

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Subscribe to an event type (sync callback).

        Args:
            event_type: Type of event to subscribe to
            callback: Callback function to call when event is published
        """
        self._subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to {event_type}")

    def subscribe_async(self, event_type: EventType, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        """Subscribe to an event type (async callback).

        Args:
            event_type: Type of event to subscribe to
            callback: Async callback function to call when event is published
        """
        self._async_subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed async to {event_type}")

    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed from {event_type}")

    def unsubscribe_async(self, event_type: EventType, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        """Unsubscribe from an event type (async).

        Args:
            event_type: Type of event to unsubscribe from
            callback: Async callback function to remove
        """
        if callback in self._async_subscribers[event_type]:
            self._async_subscribers[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed async from {event_type}")
    
    def subscribe_all_async(self, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        """Subscribe to all events (async).
        
        Useful for SSE streaming - receives all events regardless of type.
        
        Args:
            callback: Async callback function to call for all events
        """
        self._all_subscribers.append(callback)
        self.logger.debug("Subscribed to all events")
    
    def unsubscribe_all_async(self, callback: Callable[[Event], Coroutine[Any, Any, None]]):
        """Unsubscribe from all events (async).
        
        Args:
            callback: Async callback function to remove
        """
        if callback in self._all_subscribers:
            self._all_subscribers.remove(callback)
            self.logger.debug("Unsubscribed from all events")


# Global bus instance
_bus: Optional[Bus] = None


def get_bus() -> Bus:
    """Get global bus instance.

    Returns:
        Bus instance
    """
    global _bus
    if _bus is None:
        _bus = Bus()
    return _bus


# Convenience functions
async def publish(event_type: EventType, properties: Dict[str, Any]):
    """Publish an event to the global bus.

    Args:
        event_type: Type of event
        properties: Event properties
    """
    bus = get_bus()
    await bus.publish(event_type, properties)


def subscribe(event_type: EventType, callback: Callable[[Event], None]):
    """Subscribe to an event type on the global bus.

    Args:
        event_type: Type of event to subscribe to
        callback: Callback function
    """
    bus = get_bus()
    bus.subscribe(event_type, callback)


def subscribe_async(event_type: EventType, callback: Callable[[Event], Coroutine[Any, Any, None]]):
    """Subscribe to an event type on the global bus (async).

    Args:
        event_type: Type of event to subscribe to
        callback: Async callback function
    """
    bus = get_bus()
    bus.subscribe_async(event_type, callback)

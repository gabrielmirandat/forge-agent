"""Helper functions for publishing events with session existence checks.

This module provides utilities to prevent publishing events for deleted sessions,
avoiding unnecessary frontend requests and 404 errors.
"""

from typing import Any, Dict, Optional

from agent.runtime.bus import EventType, publish
from agent.storage import NotFoundError, Storage


async def session_exists(storage: Storage, session_id: str) -> bool:
    """Check if a session exists in storage.
    
    Args:
        storage: Storage instance
        session_id: Session identifier
        
    Returns:
        True if session exists, False otherwise
    """
    try:
        await storage.get_session(session_id)
        return True
    except NotFoundError:
        return False
    except Exception:
        # On any other error, assume session doesn't exist to be safe
        return False


async def publish_if_session_exists(
    event_type: EventType,
    properties: Dict[str, Any],
    storage: Optional[Storage] = None,
) -> bool:
    """Publish event only if session still exists.
    
    This prevents publishing events for deleted sessions, avoiding
    unnecessary frontend requests and 404 errors.
    
    Args:
        event_type: Type of event to publish
        properties: Event properties (must include session_id)
        storage: Optional Storage instance to check session existence.
                 If None, event is published without checking (for backward compatibility)
        
    Returns:
        True if event was published, False if session doesn't exist
    """
    session_id = properties.get("session_id")
    if not session_id:
        # No session_id, publish anyway (might be a global event)
        await publish(event_type, properties)
        return True
    
    # If no storage provided, publish anyway (backward compatibility)
    if storage is None:
        await publish(event_type, properties)
        return True
    
    # Check if session exists before publishing
    if not await session_exists(storage, session_id):
        from agent.observability import get_logger
        logger = get_logger("runtime.event_helpers", "runtime")
        logger.debug(f"Skipping event {event_type} for deleted session: {session_id}")
        return False
    
    # Session exists, publish event
    await publish(event_type, properties)
    return True

"""Persistence-level data models.

These models represent what is stored, separate from runtime schemas.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role in a chat session."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a chat session."""

    message_id: str = Field(..., description="Unique message identifier")
    session_id: str = Field(..., description="Session identifier")
    role: MessageRole = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")
    created_at: float = Field(..., description="Unix timestamp when message was created")


class Session(BaseModel):
    """A chat session with message history."""

    session_id: str = Field(..., description="Unique session identifier")
    title: str = Field(..., description="Session title (first user message or generated)")
    messages: List[Message] = Field(
        default_factory=list, description="List of messages in the session"
    )
    created_at: float = Field(..., description="Unix timestamp when session was created")
    updated_at: float = Field(..., description="Unix timestamp when session was last updated")


class SessionSummary(BaseModel):
    """Summary of a session for listing endpoints."""

    session_id: str = Field(..., description="Unique session identifier")
    title: str = Field(..., description="Session title")
    created_at: float = Field(..., description="Unix timestamp when session was created")
    updated_at: float = Field(..., description="Unix timestamp when session was last updated")
    message_count: int = Field(..., description="Number of messages in the session")

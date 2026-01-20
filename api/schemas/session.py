"""API schemas for session/chat endpoints."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Request schema for creating a new session."""

    title: Optional[str] = Field(
        default=None, description="Optional session title (auto-generated from first message if not provided)"
    )


class CreateSessionResponse(BaseModel):
    """Response schema for creating a session."""

    session_id: str = Field(..., description="Session identifier")
    title: str = Field(..., description="Session title")


class MessageRequest(BaseModel):
    """Request schema for sending a message."""

    content: str = Field(..., min_length=1, description="Message content")


class MessageResponse(BaseModel):
    """Response schema for a message."""

    message_id: str = Field(..., description="Message identifier")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    created_at: float = Field(..., description="Unix timestamp")


class SessionResponse(BaseModel):
    """Response schema for getting a session."""

    session_id: str = Field(..., description="Session identifier")
    title: str = Field(..., description="Session title")
    messages: List[MessageResponse] = Field(..., description="List of messages")
    created_at: float = Field(..., description="Unix timestamp when session was created")
    updated_at: float = Field(..., description="Unix timestamp when session was last updated")


class SessionsListResponse(BaseModel):
    """Response schema for listing sessions."""

    sessions: List[Dict[str, Any]] = Field(..., description="List of session summaries")
    limit: int = Field(..., description="Limit used")
    offset: int = Field(..., description="Offset used")


# Removed ApproveOperationsRequest, RejectOperationsRequest, ApproveOperationsResponse
# No longer needed with direct tool calling

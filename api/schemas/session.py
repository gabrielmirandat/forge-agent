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
    execution_policy: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional execution policy"
    )


class MessageResponse(BaseModel):
    """Response schema for a message."""

    message_id: str = Field(..., description="Message identifier")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    created_at: float = Field(..., description="Unix timestamp")
    plan_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Plan result if message triggered planning"
    )
    execution_result: Optional[Dict[str, Any]] = Field(
        default=None, description="Execution result if message triggered execution"
    )
    pending_approval_steps: Optional[List[int]] = Field(
        default=None, description="List of step IDs that require approval (if any)"
    )
    restricted_steps: Optional[List[int]] = Field(
        default=None, description="List of step IDs that use restricted commands (if any)"
    )


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


class ApproveOperationsRequest(BaseModel):
    """Request schema for approving operations."""

    step_ids: List[int] = Field(..., description="List of step IDs to approve")
    reason: Optional[str] = Field(default=None, description="Optional approval reason")


class RejectOperationsRequest(BaseModel):
    """Request schema for rejecting operations."""

    step_ids: List[int] = Field(..., description="List of step IDs to reject")
    reason: Optional[str] = Field(default=None, description="Optional rejection reason")


class ApproveOperationsResponse(BaseModel):
    """Response schema for approving operations."""

    message_id: str = Field(..., description="Message identifier")
    execution_result: Dict[str, Any] = Field(..., description="Execution result")

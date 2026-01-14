"""Persistence-level data models.

These models represent what is stored, separate from runtime schemas.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ApprovalStatus(str, Enum):
    """Approval status for runs."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


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
    # Optional fields for assistant messages
    plan_result: Optional[dict] = Field(
        default=None, description="Plan result if message triggered planning"
    )
    execution_result: Optional[dict] = Field(
        default=None, description="Execution result if message triggered execution"
    )


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


# Legacy models (kept for backward compatibility)
class RunRecord(BaseModel):
    """Complete run record with full plan and execution data.

    Stores raw JSON snapshots for full auditability.
    Includes approval state for Human-in-the-Loop workflows.
    """

    run_id: str = Field(..., description="Unique run identifier")
    plan_id: str = Field(..., description="Plan identifier")
    objective: str = Field(..., description="Plan objective")
    plan_result: dict = Field(..., description="Full plan result as JSON")
    execution_result: Optional[dict] = Field(
        default=None, description="Full execution result as JSON (None if not executed)"
    )
    created_at: float = Field(..., description="Unix timestamp when run was created")
    approval_status: ApprovalStatus = Field(
        default=ApprovalStatus.PENDING, description="Approval status"
    )
    approval_reason: Optional[str] = Field(
        default=None, description="Reason for approval or rejection"
    )
    approved_at: Optional[float] = Field(
        default=None, description="Unix timestamp when approved/rejected"
    )
    approved_by: Optional[str] = Field(
        default=None, description="Who approved/rejected (free-form string)"
    )


class RunSummary(BaseModel):
    """Summary of a run for listing endpoints.

    Contains only essential information for browsing runs.
    """

    run_id: str = Field(..., description="Unique run identifier")
    plan_id: str = Field(..., description="Plan identifier")
    objective: str = Field(..., description="Plan objective")
    success: bool = Field(..., description="Whether execution succeeded")
    created_at: float = Field(..., description="Unix timestamp when run was created")

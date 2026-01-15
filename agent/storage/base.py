"""Storage abstraction interface.

Storage is passive - it never mutates data or affects execution behavior.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.storage.models import ApprovalStatus, Message, MessageRole, Session, SessionSummary


class Storage(ABC):
    """Abstract storage interface for persisting plans, executions, and runs.

    Storage is passive:
    - No business logic
    - Never mutates data
    - Failures surface explicitly
    - Never affects execution behavior
    """

    # Session methods
    @abstractmethod
    async def create_session(self, title: str | None = None) -> "Session":
        """Create a new chat session.

        Args:
            title: Optional session title (auto-generated if None)

        Returns:
            Created Session

        Raises:
            StorageError: If creation fails
        """
        pass

    @abstractmethod
    async def get_session(self, session_id: str) -> "Session":
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session with all messages

        Raises:
            StorageError: If retrieval fails
            NotFoundError: If session not found
        """
        pass

    @abstractmethod
    async def add_message(
        self,
        session_id: str,
        role: "MessageRole",
        content: str,
        plan_result: dict | None = None,
        execution_result: dict | None = None,
    ) -> "Message":
        """Add a message to a session.

        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            plan_result: Optional plan result
            execution_result: Optional execution result

        Returns:
            Created Message

        Raises:
            StorageError: If save fails
            NotFoundError: If session not found
        """
        pass

    @abstractmethod
    async def list_sessions(self, limit: int = 20, offset: int = 0) -> list["SessionSummary"]:
        """List sessions with pagination.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of SessionSummary objects

        Raises:
            StorageError: If retrieval fails
        """
        pass

    @abstractmethod
    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update session title.

        Args:
            session_id: Session identifier
            title: New title

        Raises:
            StorageError: If update fails
            NotFoundError: If session not found
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all its messages.

        Args:
            session_id: Session identifier

        Raises:
            StorageError: If deletion fails
            NotFoundError: If session not found
        """
        pass

    @abstractmethod
    async def get_tmux_session(self, session_id: str) -> str | None:
        """Get tmux session name for an agent session.
        
        Args:
            session_id: Agent session ID
            
        Returns:
            Tmux session name or None if not found
            
        Raises:
            NotFoundError: If session not found
        """
        pass


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class NotFoundError(StorageError):
    """Raised when a requested resource is not found."""

    pass


"""JSON-based storage implementation - stores sessions and messages in JSON files.

Similar to OpenCode's approach: each session is stored as a JSON file.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.observability import get_logger
from agent.storage.base import Storage
from agent.storage.models import Message, MessageRole, Session, SessionSummary


class JSONStorage(Storage):
    """JSON-based storage implementation."""

    def __init__(self, storage_path: str = "~/.forge-agent/sessions"):
        """Initialize JSON storage.

        Args:
            storage_path: Base path for storing session JSON files
        """
        self.storage_path = Path(storage_path).expanduser().resolve()
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("storage.json", "storage")

    def _get_session_file(self, session_id: str) -> Path:
        """Get path to session JSON file.

        Args:
            session_id: Session identifier

        Returns:
            Path to session file
        """
        return self.storage_path / f"{session_id}.json"

    def _load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from JSON file.

        Args:
            session_id: Session identifier

        Returns:
            Session data dict or None if not found
        """
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading session {session_id}: {e}")
            return None

    def _save_session(self, session_id: str, data: Dict[str, Any]):
        """Save session data to JSON file.

        Args:
            session_id: Session identifier
            data: Session data dict
        """
        session_file = self._get_session_file(session_id)
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving session {session_id}: {e}")
            raise

    async def create_session(self, title: Optional[str] = None) -> Session:
        """Create a new session.

        Args:
            title: Optional session title

        Returns:
            Created Session
        """
        from agent.id import ascending

        session_id = ascending("session")
        now = time.time()

        # Create PTY session for this agent session
        from agent.runtime.pty_manager import get_pty_manager
        
        pty_manager = get_pty_manager()
        try:
            pty_session = pty_manager.create_session(session_id)
            pty_session_id = session_id  # PTY session uses same ID as agent session
            self.logger.info(f"Created PTY session {pty_session_id} for agent session {session_id}")
        except Exception as e:
            self.logger.warning(f"Failed to create PTY session: {e}")
            pty_session_id = None

        session_data = {
            "session_id": session_id,
            "title": title or f"Session {session_id[:8]}",
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "pty_session": pty_session_id,
        }

        self._save_session(session_id, session_data)

        self.logger.info(f"Created session {session_id}")

        return Session(
            session_id=session_id,
            title=session_data["title"],
            created_at=now,
            updated_at=now,
            messages=[],
            pty_session=pty_session_id,
        )

    async def get_session(self, session_id: str) -> Session:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session object

        Raises:
            NotFoundError: If session not found
        """
        from agent.storage import NotFoundError

        data = self._load_session(session_id)
        if not data:
            raise NotFoundError(f"Session not found: {session_id}")

        # Convert messages
        messages = []
        for msg_data in data.get("messages", []):
            messages.append(
                Message(
                    message_id=msg_data["message_id"],
                    session_id=msg_data.get("session_id", session_id),
                    role=MessageRole(msg_data["role"]),
                    content=msg_data["content"],
                    created_at=msg_data.get("created_at", time.time()),
                )
            )

        return Session(
            session_id=data["session_id"],
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            messages=messages,
            pty_session=data.get("pty_session"),
        )

    async def list_sessions(self, limit: int = 20, offset: int = 0) -> List[SessionSummary]:
        """List sessions with pagination.

        Args:
            limit: Maximum number of sessions
            offset: Number of sessions to skip

        Returns:
            List of session summaries
        """
        summaries = []

        # Get all session files
        session_files = sorted(
            self.storage_path.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for session_file in session_files[offset : offset + limit]:
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Calculate message_count from messages list
                    message_count = len(data.get("messages", []))
                    summaries.append(
                        SessionSummary(
                            session_id=data["session_id"],
                            title=data["title"],
                            created_at=data["created_at"],
                            updated_at=data["updated_at"],
                            message_count=message_count,
                        )
                    )
            except Exception as e:
                self.logger.warning(f"Error loading session from {session_file}: {e}")

        return summaries

    async def delete_session(self, session_id: str):
        """Delete a session.

        Args:
            session_id: Session identifier

        Raises:
            NotFoundError: If session not found
        """
        from agent.storage import NotFoundError

        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            raise NotFoundError(f"Session not found: {session_id}")

        # Delete PTY session if exists
        from agent.runtime.pty_manager import get_pty_manager
        
        pty_manager = get_pty_manager()
        try:
            deleted = pty_manager.delete_session(session_id)
            if deleted:
                self.logger.info(f"Deleted PTY session {session_id}")
        except Exception as e:
            self.logger.warning(f"Failed to delete PTY session: {e}")

        session_file.unlink()
        self.logger.info(f"Deleted session {session_id}")

    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
    ) -> Message:
        """Add a message to a session.

        Args:
            session_id: Session identifier
            role: Message role
            content: Message content

        Returns:
            Created Message

        Raises:
            NotFoundError: If session not found
        """
        from agent.storage import NotFoundError
        from agent.id import ascending

        data = self._load_session(session_id)
        if not data:
            raise NotFoundError(f"Session not found: {session_id}")

        message_id = ascending("message")
        now = time.time()

        message_data = {
            "message_id": message_id,
            "session_id": session_id,
            "role": role.value,
            "content": content,
            "created_at": now,
        }

        data["messages"].append(message_data)
        data["updated_at"] = now

        self._save_session(session_id, data)

        return Message(
            message_id=message_id,
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
        )

    async def update_session_title(self, session_id: str, title: str):
        """Update session title.

        Args:
            session_id: Session identifier
            title: New title

        Raises:
            NotFoundError: If session not found
        """
        from agent.storage import NotFoundError

        data = self._load_session(session_id)
        if not data:
            raise NotFoundError(f"Session not found: {session_id}")

        data["title"] = title
        data["updated_at"] = time.time()

        self._save_session(session_id, data)

    def get_pty_session(self, session_id: str) -> Optional[str]:
        """Get PTY session ID for a session.

        Args:
            session_id: Session identifier

        Returns:
            PTY session ID or None
        """
        data = self._load_session(session_id)
        if not data:
            return None

        return data.get("pty_session")

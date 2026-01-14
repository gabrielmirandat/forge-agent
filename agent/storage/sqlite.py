"""SQLite storage implementation.

Simple SQLite-based storage with auto-migration on startup.
No ORM - uses aiosqlite directly.
"""

import json
import time
import uuid
from pathlib import Path
from typing import List

import aiosqlite

from agent.observability import get_logger
from agent.storage.base import NotFoundError, Storage, StorageError
from agent.storage.models import (
    Message,
    MessageRole,
    Session,
    SessionSummary,
)


class SQLiteStorage(Storage):
    """SQLite-based storage implementation.

    Single file database with auto-migration on startup.
    JSON stored as TEXT.
    """

    def __init__(self, db_path: str | Path = "forge_agent.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._initialized = False
        self.logger = get_logger("storage", "storage")

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized and migrated."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Drop obsolete runs table if it exists
            await db.execute("DROP TABLE IF EXISTS runs")
            
            # Create sessions table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """
            )

            # Create messages table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    plan_result TEXT,
                    execution_result TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """
            )

            # Create indexes
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at 
                ON sessions(updated_at DESC)
            """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                ON messages(session_id, created_at ASC)
            """
            )

            await db.commit()

        self._initialized = True

    # Session methods
    async def create_session(self, title: str | None = None) -> Session:
        """Create a new chat session.

        Args:
            title: Optional session title (auto-generated if None)

        Returns:
            Created Session

        Raises:
            StorageError: If creation fails
        """
        await self._ensure_initialized()

        session_id = f"session-{uuid.uuid4().hex[:12]}"
        now = time.time()
        final_title = title or "New Chat"

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO sessions (session_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (session_id, final_title, now, now),
                )
                await db.commit()

            return Session(
                session_id=session_id,
                title=final_title,
                messages=[],
                created_at=now,
                updated_at=now,
            )
        except Exception as e:
            raise StorageError(f"Failed to create session: {e}") from e

    async def get_session(self, session_id: str) -> Session:
        """Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session with all messages

        Raises:
            StorageError: If retrieval fails
            NotFoundError: If session not found
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                # Get session
                async with db.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ) as cursor:
                    session_row = await cursor.fetchone()
                    if session_row is None:
                        raise NotFoundError(f"Session not found: {session_id}")

                # Get messages
                async with db.execute(
                    """
                    SELECT * FROM messages 
                    WHERE session_id = ? 
                    ORDER BY created_at ASC
                """,
                    (session_id,),
                ) as cursor:
                    message_rows = await cursor.fetchall()
                    messages = []
                    for row in message_rows:
                        plan_result = None
                        if row["plan_result"]:
                            plan_result = json.loads(row["plan_result"])
                        execution_result = None
                        if row["execution_result"]:
                            execution_result = json.loads(row["execution_result"])

                        messages.append(
                            Message(
                                message_id=row["message_id"],
                                session_id=row["session_id"],
                                role=MessageRole(row["role"]),
                                content=row["content"],
                                created_at=row["created_at"],
                                plan_result=plan_result,
                                execution_result=execution_result,
                            )
                        )

                return Session(
                    session_id=session_row["session_id"],
                    title=session_row["title"],
                    messages=messages,
                    created_at=session_row["created_at"],
                    updated_at=session_row["updated_at"],
                )
        except NotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to get session: {e}") from e

    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        plan_result: dict | None = None,
        execution_result: dict | None = None,
    ) -> Message:
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
        await self._ensure_initialized()

        message_id = f"msg-{uuid.uuid4().hex[:12]}"
        now = time.time()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Check if session exists
                async with db.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
                ) as cursor:
                    if await cursor.fetchone() is None:
                        raise NotFoundError(f"Session not found: {session_id}")

                # Insert message
                await db.execute(
                    """
                    INSERT INTO messages (message_id, session_id, role, content, created_at, plan_result, execution_result)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        message_id,
                        session_id,
                        role.value,
                        content,
                        now,
                        json.dumps(plan_result) if plan_result else None,
                        json.dumps(execution_result) if execution_result else None,
                    ),
                )

                # Update session updated_at
                await db.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )

                await db.commit()

            return Message(
                message_id=message_id,
                session_id=session_id,
                role=role,
                content=content,
                created_at=now,
                plan_result=plan_result,
                execution_result=execution_result,
            )
        except NotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to add message: {e}") from e

    async def list_sessions(self, limit: int = 20, offset: int = 0) -> List[SessionSummary]:
        """List sessions with pagination.

        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of SessionSummary objects

        Raises:
            StorageError: If retrieval fails
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    """
                    SELECT s.session_id, s.title, s.created_at, s.updated_at,
                           COUNT(m.message_id) as message_count
                    FROM sessions s
                    LEFT JOIN messages m ON s.session_id = m.session_id
                    GROUP BY s.session_id
                    ORDER BY s.updated_at DESC
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                ) as cursor:
                    rows = await cursor.fetchall()
                    summaries = []
                    for row in rows:
                        summaries.append(
                            SessionSummary(
                                session_id=row["session_id"],
                                title=row["title"],
                                created_at=row["created_at"],
                                updated_at=row["updated_at"],
                                message_count=row["message_count"] or 0,
                            )
                        )
                    return summaries
        except Exception as e:
            raise StorageError(f"Failed to list sessions: {e}") from e

    async def update_session_title(self, session_id: str, title: str) -> None:
        """Update session title.

        Args:
            session_id: Session identifier
            title: New title

        Raises:
            StorageError: If update fails
            NotFoundError: If session not found
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Check if session exists
                async with db.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
                ) as cursor:
                    if await cursor.fetchone() is None:
                        raise NotFoundError(f"Session not found: {session_id}")

                # Update title
                await db.execute(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    (title, session_id),
                )
                await db.commit()
        except NotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to update session title: {e}") from e

    async def delete_session(self, session_id: str) -> None:
        """Delete a session and all its messages.

        Args:
            session_id: Session identifier

        Raises:
            StorageError: If deletion fails
            NotFoundError: If session not found
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Check if session exists
                async with db.execute(
                    "SELECT session_id FROM sessions WHERE session_id = ?", (session_id,)
                ) as cursor:
                    if await cursor.fetchone() is None:
                        raise NotFoundError(f"Session not found: {session_id}")

                # Delete messages first (foreign key constraint)
                await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                
                # Delete session
                await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                
                await db.commit()
        except NotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to delete session: {e}") from e

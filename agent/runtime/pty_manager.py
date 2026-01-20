"""PTY Manager - manages pseudo-terminal sessions for command execution.

Replaces tmux with PTY-based approach similar to OpenCode.
Each session maintains its own working directory and environment.
"""

import asyncio
import os
import pty
import select
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from agent.observability import get_logger


class PTYManager:
    """Manages PTY sessions for command execution with explicit cwd."""

    def __init__(self):
        """Initialize PTY manager."""
        self.sessions: Dict[str, "PTYSession"] = {}
        self.logger = get_logger("pty", "pty")

    def create_session(
        self,
        session_id: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "PTYSession":
        """Create a new PTY session.

        Args:
            session_id: Unique session identifier
            cwd: Working directory (defaults to ~/repos)
            env: Environment variables to set

        Returns:
            PTYSession instance
        """
        if session_id in self.sessions:
            self.logger.warning(f"Session {session_id} already exists, reusing")
            return self.sessions[session_id]

        # Default cwd
        if cwd is None:
            cwd = str(Path.home() / "repos")
        else:
            cwd = str(Path(cwd).expanduser().resolve())

        # Ensure directory exists
        Path(cwd).mkdir(parents=True, exist_ok=True)

        # Merge environment
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        full_env["TERM"] = "xterm-256color"

        session = PTYSession(session_id, cwd, full_env, self.logger)
        self.sessions[session_id] = session

        self.logger.info(f"Created PTY session {session_id} with cwd={cwd}")
        return session

    def get_session(self, session_id: str) -> Optional["PTYSession"]:
        """Get an existing PTY session.

        Args:
            session_id: Session identifier

        Returns:
            PTYSession if exists, None otherwise
        """
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a PTY session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        session.close()
        del self.sessions[session_id]

        self.logger.info(f"Deleted PTY session {session_id}")
        return True

    def get_working_directory(self, session_id: str) -> Optional[str]:
        """Get current working directory of a session.

        Args:
            session_id: Session identifier

        Returns:
            Current working directory or None if session doesn't exist
        """
        session = self.sessions.get(session_id)
        if session:
            return session.cwd
        return None


class PTYSession:
    """A single PTY session with its own working directory."""

    def __init__(self, session_id: str, cwd: str, env: Dict[str, str], logger):
        """Initialize PTY session.

        Args:
            session_id: Session identifier
            cwd: Working directory
            env: Environment variables
            logger: Logger instance
        """
        self.session_id = session_id
        self.cwd = cwd
        self.env = env
        self.logger = logger
        self.process: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None

    async def execute_command(
        self,
        command: str,
        timeout: float = 30.0,
    ) -> Tuple[int, str, str]:
        """Execute a command in this PTY session with explicit cwd.

        Args:
            command: Command to execute
            timeout: Command timeout in seconds

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        # Use subprocess with explicit cwd (like OpenCode)
        # This ensures each command runs in the correct directory
        try:
            process = await asyncio.create_subprocess_exec(
                *["/bin/bash", "-c", command],
                cwd=self.cwd,
                env=self.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                return_code = process.returncode or 0

                stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
                stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

                return (return_code, stdout_text, stderr_text)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return (124, "", f"Command timed out after {timeout} seconds")
        except Exception as e:
            self.logger.error(f"Error executing command: {e}", exc_info=True)
            return (1, "", f"Error executing command: {e}")

    def change_directory(self, new_cwd: str) -> bool:
        """Change working directory for this session.

        Args:
            new_cwd: New working directory

        Returns:
            True if successful, False otherwise
        """
        try:
            resolved = str(Path(new_cwd).expanduser().resolve())
            if not Path(resolved).is_dir():
                return False
            self.cwd = resolved
            self.logger.info(f"Changed cwd to {resolved} for session {self.session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error changing directory: {e}")
            return False

    def close(self):
        """Close this PTY session."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except Exception:
                pass
            self.master_fd = None

        if self.slave_fd is not None:
            try:
                os.close(self.slave_fd)
            except Exception:
                pass
            self.slave_fd = None


# Global PTY manager instance
_pty_manager: Optional[PTYManager] = None


def get_pty_manager() -> PTYManager:
    """Get global PTY manager instance.

    Returns:
        PTYManager instance
    """
    global _pty_manager
    if _pty_manager is None:
        _pty_manager = PTYManager()
    return _pty_manager

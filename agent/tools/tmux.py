"""Tmux session management for maintaining execution context.

Each agent session has an associated tmux session that maintains
the working directory and environment state across tool executions.
"""

import asyncio
from pathlib import Path
from typing import Optional

import libtmux
from agent.observability import get_logger


class TmuxManager:
    """Manages tmux sessions for agent sessions using libtmux."""

    def __init__(self):
        """Initialize tmux manager."""
        self.logger = get_logger("tmux", "tmux")
        self._server: Optional[libtmux.Server] = None

    def _get_server(self) -> libtmux.Server:
        """Get or create libtmux Server instance.
        
        Returns:
            libtmux.Server instance
        """
        if self._server is None:
            self._server = libtmux.Server()
        return self._server

    def _get_session_name(self, session_id: str) -> str:
        """Get tmux session name from agent session ID.
        
        Args:
            session_id: Agent session ID
            
        Returns:
            Tmux session name (sanitized for tmux)
        """
        # Tmux session names must be alphanumeric, dash, or underscore
        # Replace invalid characters with underscore
        sanitized = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return f"forge-{sanitized}"

    async def create_session(self, session_id: str, initial_cwd: Optional[str] = None) -> str:
        """Create a new tmux session for an agent session.
        
        Args:
            session_id: Agent session ID
            initial_cwd: Initial working directory (default: ~/repos)
            
        Returns:
            Tmux session name
            
        Raises:
            RuntimeError: If tmux session creation fails
        """
        session_name = self._get_session_name(session_id)
        
        # Default to ~/repos if no initial_cwd specified
        if initial_cwd is None:
            initial_cwd = str(Path.home() / "repos")
        else:
            initial_cwd = str(Path(initial_cwd).expanduser().resolve())
        
        # Run in executor to avoid blocking
        def _create():
            server = self._get_server()
            
            # Check if session already exists
            existing = server.sessions.filter(session_name=session_name)
            if existing:
                self.logger.warning(f"Tmux session {session_name} already exists, reusing it")
                return session_name
            
            # Create new detached tmux session
            # libtmux handles the creation
            session = server.new_session(
                session_name=session_name,
                start_directory=initial_cwd,
                detach=True,
            )
            
            self.logger.info(f"Successfully created tmux session '{session_name}' in directory '{initial_cwd}' for agent session '{session_id}'")
            return session_name
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _create)
        except Exception as e:
            self.logger.error(f"Failed to create tmux session '{session_name}' for agent session '{session_id}': {e}", exc_info=True)
            raise RuntimeError(f"Failed to create tmux session: {e}") from e

    async def delete_session(self, session_name: str) -> bool:
        """Delete a tmux session.
        
        Args:
            session_name: Tmux session name
            
        Returns:
            True if session was deleted, False if it didn't exist
        """
        def _delete():
            server = self._get_server()
            sessions = server.sessions.filter(session_name=session_name)
            if not sessions:
                self.logger.debug(f"Tmux session {session_name} does not exist, skipping deletion")
                return False
            
            try:
                # Use kill() instead of kill_session() (deprecated in libtmux 0.30.0+)
                sessions[0].kill()
                self.logger.info(f"Successfully deleted tmux session '{session_name}'")
                return True
            except Exception as e:
                self.logger.error(f"Error deleting tmux session '{session_name}': {e}", exc_info=True)
                return False
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _delete)
        except Exception:
            return False

    async def session_exists(self, session_name: str) -> bool:
        """Check if a tmux session exists.
        
        Args:
            session_name: Tmux session name
            
        Returns:
            True if session exists, False otherwise
        """
        def _check():
            server = self._get_server()
            return bool(server.sessions.filter(session_name=session_name))
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _check)
        except Exception:
            return False

    async def execute_command(
        self,
        session_name: str,
        command: str,
        timeout: float = 30.0,
    ) -> tuple[int, str, str]:
        """Execute a command in a tmux session using send_keys.
        
        Executes the command directly in the tmux pane using send_keys,
        then captures the output from the pane. This maintains the tmux
        session's context (working directory, environment, etc.).
        
        Args:
            session_name: Tmux session name
            command: Command to execute
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        def _execute():
            server = self._get_server()
            sessions = server.sessions.filter(session_name=session_name)
            if not sessions:
                return (1, "", f"Tmux session {session_name} not found")
            
            session = sessions[0]
            window = session.active_window
            pane = window.active_pane
            
            # Clear pane to get clean output
            pane.clear()
            
            # Send command to pane
            pane.send_keys(command, enter=True)
            
            # Wait for command to execute and capture output
            import time
            start_time = time.time()
            max_wait = timeout
            
            # Poll for output - wait for command to complete
            # We check if the pane content has changed and stabilized
            last_output = ""
            stable_count = 0
            required_stable = 3  # Command is done if output is stable for 3 checks
            
            while time.time() - start_time < max_wait:
                time.sleep(0.3)  # Wait a bit between checks
                
                # Capture pane content
                try:
                    result = pane.cmd('capture-pane', '-p')
                    if result.stdout:
                        current_output = '\n'.join(result.stdout).strip()
                        
                        # Check if output has stabilized (command finished)
                        if current_output == last_output:
                            stable_count += 1
                            if stable_count >= required_stable:
                                # Output is stable, command likely finished
                                break
                        else:
                            stable_count = 0
                            last_output = current_output
                except Exception:
                    pass
            
            # Extract command output (remove command itself and prompt)
            try:
                result = pane.cmd('capture-pane', '-p')
                if result.stdout:
                    pane_lines = result.stdout
                    
                    # Find the command in the output and extract what comes after
                    output_lines = []
                    found_command = False
                    for line in pane_lines:
                        line_stripped = line.strip()
                        # Skip empty lines and prompt lines
                        if not line_stripped or line_stripped.startswith('$') or line_stripped.startswith('#'):
                            if found_command:
                                # We've seen the command, now we're past it
                                continue
                            continue
                        
                        # Check if this line contains our command
                        if command in line_stripped:
                            found_command = True
                            continue
                        
                        # After command, collect output lines
                        if found_command:
                            output_lines.append(line_stripped)
                    
                    stdout_text = '\n'.join(output_lines).strip()
                    
                    # Try to determine return code by checking for error patterns
                    return_code = 0
                    if any(keyword in stdout_text.lower() for keyword in ['error', 'failed', 'command not found', 'no such file']):
                        return_code = 1
                    
                    return (return_code, stdout_text, "")
            except Exception as e:
                return (1, "", f"Failed to capture output: {e}")
        
        try:
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, _execute),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return (124, "", f"Command timed out after {timeout} seconds")
        except Exception as e:
            return (1, "", f"Error executing command in tmux: {e}")

    async def get_working_directory(self, session_name: str) -> Optional[str]:
        """Get current working directory of tmux session.
        
        Uses libtmux to get the pane's current path.
        
        Args:
            session_name: Tmux session name
            
        Returns:
            Current working directory or None if error
        """
        def _get_cwd():
            server = self._get_server()
            sessions = server.sessions.filter(session_name=session_name)
            if not sessions:
                return None
            
            session = sessions[0]
            window = session.active_window
            pane = window.active_pane
            
            # Get current path from pane
            try:
                # Use display-message to get pane current path
                result = pane.cmd('display-message', '-p', '#{pane_current_path}')
                if result.stdout:
                    path = result.stdout[0].strip()
                    # Expand ~ and resolve to absolute path
                    return str(Path(path).expanduser().resolve())
            except Exception:
                pass
            
            return None
        
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _get_cwd)
        except Exception:
            return None


# Global instance
_tmux_manager: Optional[TmuxManager] = None


def get_tmux_manager() -> TmuxManager:
    """Get global tmux manager instance.
    
    Returns:
        TmuxManager instance
    """
    global _tmux_manager
    if _tmux_manager is None:
        _tmux_manager = TmuxManager()
    return _tmux_manager

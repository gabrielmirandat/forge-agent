"""Frontend server management for smoke tests."""

import subprocess
import sys
import time
from pathlib import Path

import httpx

from smoke_test.config import (
    FRONTEND_DIR,
    FRONTEND_PORT,
    FRONTEND_STARTUP_TIMEOUT,
    FRONTEND_URL,
)


class FrontendServer:
    """Manages frontend server lifecycle."""

    def __init__(self):
        """Initialize frontend server manager."""
        self.process = None
        self.started = False

    def start(self) -> None:
        """Start the frontend server.

        Raises:
            RuntimeError: If server fails to start within timeout
        """
        if self.started:
            return

        # Check if node_modules exists
        node_modules = FRONTEND_DIR / "node_modules"
        if not node_modules.exists():
            raise RuntimeError(
                "Frontend node_modules not found. Run 'npm install' in frontend directory first."
            )

        print(f"[Frontend] Starting server on port {FRONTEND_PORT}...")

        # Start npm run dev
        self.process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for frontend to be accessible
        start_time = time.time()
        while time.time() - start_time < FRONTEND_STARTUP_TIMEOUT:
            try:
                response = httpx.get(FRONTEND_URL, timeout=2, follow_redirects=True)
                if response.status_code == 200:
                    self.started = True
                    print(f"[Frontend] Server started successfully")
                    return
            except (httpx.RequestError, httpx.TimeoutException):
                time.sleep(0.5)

        # If we get here, startup failed
        self.stop()
        raise RuntimeError(
            f"Frontend failed to start within {FRONTEND_STARTUP_TIMEOUT} seconds"
        )

    def stop(self) -> None:
        """Stop the frontend server."""
        if self.process:
            print("[Frontend] Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.started = False
            print("[Frontend] Server stopped")

    def is_running(self) -> bool:
        """Check if frontend is running.

        Returns:
            True if frontend responds
        """
        try:
            response = httpx.get(FRONTEND_URL, timeout=2, follow_redirects=True)
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False


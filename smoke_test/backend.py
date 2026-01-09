"""Backend server management for smoke tests."""

import subprocess
import sys
import time
from pathlib import Path

import httpx

from smoke_test.config import BACKEND_PORT, BACKEND_STARTUP_TIMEOUT, BACKEND_URL, BACKEND_DIR


class BackendServer:
    """Manages backend server lifecycle."""

    def __init__(self):
        """Initialize backend server manager."""
        self.process = None
        self.started = False

    def start(self) -> None:
        """Start the backend server.

        Raises:
            RuntimeError: If server fails to start within timeout
        """
        if self.started:
            return

        print(f"[Backend] Starting server on port {BACKEND_PORT}...")

        # Start uvicorn
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(BACKEND_PORT),
            ],
            cwd=str(BACKEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for health check
        start_time = time.time()
        while time.time() - start_time < BACKEND_STARTUP_TIMEOUT:
            try:
                response = httpx.get(f"{BACKEND_URL}/health", timeout=2)
                if response.status_code == 200:
                    self.started = True
                    print(f"[Backend] Server started successfully")
                    return
            except (httpx.RequestError, httpx.TimeoutException):
                time.sleep(0.5)

        # If we get here, startup failed
        self.stop()
        raise RuntimeError(
            f"Backend failed to start within {BACKEND_STARTUP_TIMEOUT} seconds"
        )

    def stop(self) -> None:
        """Stop the backend server."""
        if self.process:
            print("[Backend] Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.started = False
            print("[Backend] Server stopped")

    def is_running(self) -> bool:
        """Check if backend is running.

        Returns:
            True if backend responds to health check
        """
        try:
            response = httpx.get(f"{BACKEND_URL}/health", timeout=2)
            return response.status_code == 200
        except (httpx.RequestError, httpx.TimeoutException):
            return False


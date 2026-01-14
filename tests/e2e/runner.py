"""E2E test runner and setup utilities."""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx


class E2ETestRunner:
    """Runner for E2E tests with backend/frontend management."""

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        frontend_url: str = "http://localhost:3000",
        backend_port: int = 8000,
        frontend_port: int = 3000,
        headless: Optional[bool] = None,
    ):
        """Initialize test runner.

        Args:
            backend_url: Backend API URL
            frontend_url: Frontend URL
            backend_port: Backend port
            frontend_port: Frontend port
            headless: If False, browser will be visible (for debugging).
                     If None, reads from E2E_HEADLESS env var or pytest option.
        """
        self.backend_url = backend_url
        self.frontend_url = frontend_url
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        
        # Determine headless mode
        if headless is None:
            headless_env = os.getenv("E2E_HEADLESS", "true")
            self.headless = headless_env.lower() == "true"
        else:
            self.headless = headless
            
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        self.project_root = Path(__file__).parent.parent.parent

    def start_backend(self, timeout: int = 30) -> None:
        """Start backend server.

        Args:
            timeout: Timeout in seconds
        """
        if self.backend_process:
            return  # Already started

        print("[E2E] Starting backend...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)

        self.backend_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(self.backend_port),
            ],
            cwd=self.project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for backend to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(f"{self.backend_url}/health")
                    if response.status_code == 200:
                        print("[E2E] Backend started successfully")
                        return
            except Exception:
                pass
            time.sleep(0.5)

        raise TimeoutError(f"Backend did not start within {timeout} seconds")

    def stop_backend(self) -> None:
        """Stop backend server."""
        if self.backend_process:
            print("[E2E] Stopping backend...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            self.backend_process = None

    def start_frontend(self, timeout: int = 30) -> None:
        """Start frontend server.

        Args:
            timeout: Timeout in seconds
        """
        if self.frontend_process:
            return  # Already started

        print("[E2E] Starting frontend...")
        frontend_dir = self.project_root / "frontend"

        self.frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for frontend to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with httpx.Client(timeout=5.0) as client:
                    response = client.get(self.frontend_url)
                    if response.status_code == 200:
                        print("[E2E] Frontend started successfully")
                        return
            except Exception:
                pass
            time.sleep(0.5)

        raise TimeoutError(f"Frontend did not start within {timeout} seconds")

    def stop_frontend(self) -> None:
        """Stop frontend server."""
        if self.frontend_process:
            print("[E2E] Stopping frontend...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            self.frontend_process = None

    def cleanup_database(self) -> None:
        """Clean up test database."""
        db_file = self.project_root / "forge_agent.db"
        if db_file.exists():
            db_file.unlink()
            print("[E2E] Cleaned up database")

    def cleanup_logs(self) -> None:
        """Clean up test logs."""
        log_dir = self.project_root / "workspace" / "logs"
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                log_file.unlink()
        audit_log = self.project_root / "workspace" / "audit.log"
        if audit_log.exists():
            audit_log.unlink()
        print("[E2E] Cleaned up logs")

    def setup(self) -> None:
        """Set up test environment."""
        # Note: Database cleanup is done once at session start, not per test
        # This allows runs to persist across tests in the same session
        self.cleanup_logs()
        self.start_backend()
        self.start_frontend()  # Frontend required for E2E browser tests

    def teardown(self) -> None:
        """Tear down test environment."""
        self.stop_backend()
        self.stop_frontend()

    def __enter__(self):
        """Async context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.teardown()

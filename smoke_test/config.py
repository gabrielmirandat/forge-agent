"""Configuration for smoke tests."""

import os
from pathlib import Path

# Ports
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))

# URLs
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

# Timeouts (seconds)
BACKEND_STARTUP_TIMEOUT = 30
FRONTEND_STARTUP_TIMEOUT = 30
API_REQUEST_TIMEOUT = 60
BROWSER_TIMEOUT = 10

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_DIR = PROJECT_ROOT
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATABASE_PATH = PROJECT_ROOT / "forge_agent.db"

# Test goals
SUCCESS_GOAL = "List files in the current directory"
FAILURE_GOAL = "Delete a file that does not exist: /nonexistent/file.txt"

# Browser settings
HEADLESS = False  # Must be False for visual validation
BROWSER_TYPE = "chromium"  # chromium, firefox, webkit


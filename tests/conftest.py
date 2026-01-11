"""Pytest configuration and E2E test options.

This module provides:
- pytest_addoption: Custom command-line options (--headless, --browser-delay)
- pytest_configure: Environment variable setup for E2E tests
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
# TODO: Technical debt - Replace with proper package installation (pip install -e .)
# This hack can mask import errors and break in CI/production
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--headless",
        action="store",
        default=None,  # Changed: None means not provided
        help="Run browser in headless mode (true/false). If not provided, uses E2E_HEADLESS env var.",
    )
    parser.addoption(
        "--browser-delay",
        action="store",
        default=None,  # Changed: None means not provided
        type=int,
        help="Delay in seconds before closing browser after last test. If not provided, uses E2E_BROWSER_DELAY env var.",
    )

def pytest_configure(config):
    """Configure pytest with custom options."""
    # Store headless option for E2E tests
    # Priority: --headless option > E2E_HEADLESS env var (if already set) > default "true"
    headless_option = config.getoption("--headless", default=None)
    
    # Only set if option was explicitly provided OR if env var doesn't exist
    if headless_option is not None:
        # Option provided - use it
        os.environ["E2E_HEADLESS"] = headless_option
    elif "E2E_HEADLESS" not in os.environ:
        # No option and no env var - use default
        os.environ["E2E_HEADLESS"] = "true"
    # If E2E_HEADLESS already exists in env, keep it (don't override)
    
    # Store browser delay
    # Default to 0 for CI (no delay), override locally with --browser-delay N
    browser_delay_option = config.getoption("--browser-delay", default=None)
    if browser_delay_option is not None:
        os.environ["E2E_BROWSER_DELAY"] = str(browser_delay_option)
    elif "E2E_BROWSER_DELAY" not in os.environ:
        os.environ["E2E_BROWSER_DELAY"] = "0"  # CI-friendly default (no delay)


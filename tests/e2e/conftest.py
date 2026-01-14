"""Pytest fixtures for E2E tests."""

import os
import time
from pathlib import Path

import pytest

from tests.e2e.browser import BrowserHelper


@pytest.fixture(scope="session", autouse=True)
def cleanup_database_once():
    """Clean up database once at the start of the test session.
    
    This ensures the database is clean before all tests run, but runs
    created during tests persist across tests in the same session.
    """
    import sys
    project_root = Path(__file__).parent.parent.parent
    db_file = project_root / "forge_agent.db"
    
    if db_file.exists():
        db_file.unlink()
        print("[FIXTURE] Cleaned up database at session start", file=sys.stderr, flush=True)
    
    # Also clean logs at session start
    log_dir = project_root / "workspace" / "logs"
    if log_dir.exists():
        for log_file in log_dir.glob("*.log"):
            log_file.unlink()
    audit_log = project_root / "workspace" / "audit.log"
    if audit_log.exists():
        audit_log.unlink()
    print("[FIXTURE] Cleaned up logs at session start", file=sys.stderr, flush=True)
    
    yield
    
    # Optional: Clean up at session end (commented out to preserve data for debugging)
    # if db_file.exists():
    #     db_file.unlink()
    #     print("[FIXTURE] Cleaned up database at session end", file=sys.stderr, flush=True)


@pytest.fixture(scope="session")
def browser_headless():
    """Get headless setting from environment or pytest option."""
    import sys
    headless_env = os.getenv("E2E_HEADLESS", "true")
    result = headless_env.lower() == "true"
    print(f"[FIXTURE] E2E_HEADLESS={headless_env}, browser_headless={result}", file=sys.stderr, flush=True)
    return result


@pytest.fixture(scope="session")
def browser_delay():
    """Get browser delay from environment or pytest option."""
    return int(os.getenv("E2E_BROWSER_DELAY", "0"))  # CI-friendly default


@pytest.fixture(scope="session")
def shared_browser(browser_headless, browser_delay):
    """Shared browser instance for all E2E tests.
    
    Browser opens once at the start of the test session and closes
    after all tests complete (with optional delay).
    
    Note: E2E tests should be marked with @pytest.mark.e2e to allow filtering:
    pytest -m e2e
    """
    import sys
    
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # Start browser
    print(f"[FIXTURE] Creating BrowserHelper with headless={browser_headless}", file=sys.stderr, flush=True)
    browser = BrowserHelper(headless=browser_headless, frontend_url=frontend_url)
    
    # Print to stderr so it's visible even if stdout is buffered
    if not browser_headless:
        print(f"\n[E2E] ========================================", file=sys.stderr, flush=True)
        print(f"[E2E] Opening browser in VISIBLE mode...", file=sys.stderr, flush=True)
        print(f"[E2E] Browser will stay open for all tests", file=sys.stderr, flush=True)
        print(f"[E2E] Browser will close automatically after {browser_delay}s delay at the end", file=sys.stderr, flush=True)
        print(f"[E2E] ========================================\n", file=sys.stderr, flush=True)
    else:
        print(f"[E2E] Browser will run in HEADLESS mode", file=sys.stderr, flush=True)
    
    print(f"[FIXTURE] Starting browser...", file=sys.stderr, flush=True)
    browser.start()
    
    # Verify browser is ready
    if not browser.page:
        raise RuntimeError("Browser page not created after start()")
    
    print(f"[FIXTURE] Browser ready. Page URL: {browser.page.url if browser.page else 'N/A'}", file=sys.stderr, flush=True)
    
    if not browser_headless:
        print(f"[E2E] ✅ Browser opened successfully! Window should be visible now.", file=sys.stderr, flush=True)
    else:
        print(f"[E2E] Browser started in headless mode", file=sys.stderr, flush=True)
    
    yield browser
    
    print(f"[FIXTURE] Test session finished, cleaning up browser...", file=sys.stderr, flush=True)
    
    # Close browser after all tests
    if not browser_headless:
        print(f"\n[E2E] All tests completed. Browser will close in {browser_delay} seconds...", file=sys.stderr, flush=True)
        time.sleep(browser_delay)
    
    try:
        browser.stop()
    except Exception as e:
        # Ignore errors when closing (browser might already be closed)
        if not browser_headless:
            print(f"[E2E] Error closing browser (ignored): {e}", file=sys.stderr, flush=True)
    
    if not browser_headless:
        print("[E2E] Browser closed\n", file=sys.stderr, flush=True)


@pytest.fixture(autouse=True, scope="function")
def delay_between_tests(browser_headless):
    """Add delay between tests when headless=false for visibility."""
    yield
    # Delay after test completes (before next test starts)
    if not browser_headless:
        import sys
        print(f"[FIXTURE] Test completed. Waiting {BrowserHelper.VISIBLE_MODE_TEST_DELAY_MS}ms before next test...", file=sys.stderr, flush=True)
        time.sleep(BrowserHelper.VISIBLE_MODE_TEST_DELAY_MS / 1000.0)

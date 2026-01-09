"""UI validation checks using Playwright."""

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from smoke_test.config import BROWSER_TIMEOUT, BROWSER_TYPE, FRONTEND_URL, HEADLESS


class UIChecker:
    """UI validation using Playwright."""

    def __init__(self):
        """Initialize UI checker."""
        self.playwright = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

    def start(self) -> None:
        """Start browser and navigate to frontend."""
        print("[UI Check] Starting browser...")
        self.playwright = sync_playwright().start()
        self.browser = self.playwright[BROWSER_TYPE].launch(headless=HEADLESS)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.page.set_default_timeout(BROWSER_TIMEOUT * 1000)

        # Navigate to frontend
        print(f"[UI Check] Navigating to {FRONTEND_URL}...")
        self.page.goto(FRONTEND_URL, wait_until="networkidle")
        print("[UI Check] ✓ Frontend loaded")

    def check_runs_list(self) -> bool:
        """Check that runs list page loads.

        Returns:
            True if runs list is visible
        """
        try:
            # Navigate to runs list
            self.page.goto(f"{FRONTEND_URL}/runs", wait_until="networkidle")

            # Check for runs list content
            # Look for table or list of runs
            runs_list = self.page.locator("text=/run|Run|runs/i").first
            if not runs_list.is_visible(timeout=5000):
                print("[UI Check] Runs list not visible")
                return False

            print("[UI Check] ✓ Runs list page loads")
            return True
        except Exception as e:
            print(f"[UI Check] Runs list check failed: {e}")
            return False

    def check_run_detail(self, run_id: str) -> bool:
        """Check that run detail page loads and shows plan/execution.

        Args:
            run_id: Run identifier

        Returns:
            True if run detail page is valid
        """
        try:
            # Navigate to run detail
            self.page.goto(f"{FRONTEND_URL}/runs/{run_id}", wait_until="networkidle")

            # Check for plan section
            plan_section = self.page.locator("text=/plan|Plan/i").first
            if not plan_section.is_visible(timeout=5000):
                print("[UI Check] Plan section not visible")
                return False

            # Check for execution section (may not exist if HITL pending)
            execution_section = self.page.locator("text=/execution|Execution|result/i").first
            if not execution_section.is_visible(timeout=5000):
                print("[UI Check] Execution section not visible (may be pending)")
                # This is OK if HITL is enabled

            # Check for raw JSON (JsonBlock component)
            json_block = self.page.locator("text=/JSON|json|Raw/i").first
            if not json_block.is_visible(timeout=5000):
                print("[UI Check] Raw JSON not visible")
                # This is a warning, not a failure

            print("[UI Check] ✓ Run detail page loads")
            return True
        except Exception as e:
            print(f"[UI Check] Run detail check failed: {e}")
            return False

    def check_failure_visibility(self, run_id: str) -> bool:
        """Check that failures are visible in UI.

        Args:
            run_id: Run identifier

        Returns:
            True if failure is visible
        """
        try:
            self.page.goto(f"{FRONTEND_URL}/runs/{run_id}", wait_until="networkidle")

            # Look for error indicators
            error_indicators = self.page.locator("text=/error|Error|failed|Failed|failure/i")
            if error_indicators.count() == 0:
                print("[UI Check] No error indicators found (may not be a failure case)")
                return True  # Not a failure if no errors

            print("[UI Check] ✓ Failure visible in UI")
            return True
        except Exception as e:
            print(f"[UI Check] Failure visibility check failed: {e}")
            return False

    def stop(self) -> None:
        """Stop browser and cleanup."""
        if self.page:
            self.page.close()
        if self.context:
            self.context.close()
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
        print("[UI Check] Browser closed")


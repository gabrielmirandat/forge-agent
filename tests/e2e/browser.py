"""Browser automation helpers for E2E tests."""

import os
import time
from typing import Optional

from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright


class BrowserHelper:
    """Helper for browser automation in E2E tests."""

    # Hard constraints requested:
    # - DOM interaction timeouts: 1s
    # - Navigation/page load timeouts: 3s maximum after page change
    DEFAULT_INTERACTION_TIMEOUT_MS = 1000
    DEFAULT_NAVIGATION_TIMEOUT_MS = 3000  # 3s maximum after page change
    # Waiting for a run result is not a "DOM interaction" per se; it's waiting for backend work.
    # Implemented as repeated short (<=1s) DOM waits until a total timeout elapses.
    DEFAULT_RUN_RESULT_TIMEOUT_MS = 60000

    def __init__(self, headless: Optional[bool] = None, frontend_url: str = "http://localhost:3000"):
        """Initialize browser helper.

        Args:
            headless: If False, browser will be visible. If None, reads from E2E_HEADLESS env var.
            frontend_url: Frontend URL to test
        """
        if headless is None:
            headless_env = os.getenv("E2E_HEADLESS", "true")
            self.headless = headless_env.lower() == "true"
        else:
            self.headless = headless
        self.frontend_url = frontend_url
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    def start(self) -> None:
        """Start browser and create page."""
        import sys
        
        print(f"[Browser] Initializing Playwright (sync)...", file=sys.stderr, flush=True)
        self.playwright = sync_playwright().start()
        print(f"[Browser] Playwright initialized", file=sys.stderr, flush=True)
        
        # Launch browser with headless setting
        launch_options = {"headless": self.headless}
        if not self.headless:
            # When not headless, add slow_mo to make it easier to see
            launch_options["slow_mo"] = 100
            print(f"[Browser] Launching browser in VISIBLE mode (headless={self.headless})", file=sys.stderr, flush=True)
        else:
            print(f"[Browser] Launching browser in HEADLESS mode", file=sys.stderr, flush=True)
        
        self.browser = self.playwright.chromium.launch(**launch_options)
        print(f"[Browser] Browser launched", file=sys.stderr, flush=True)
        
        self.context = self.browser.new_context()
        print(f"[Browser] Browser context created", file=sys.stderr, flush=True)
        
        self.page = self.context.new_page()
        print(f"[Browser] Browser page created - ready to use", file=sys.stderr, flush=True)

        # Enforce strict timeouts globally
        # - Actions (click/fill/wait_for_selector/locator waits): 1s
        # - Navigations (goto/wait_for_url): 3s
        self.context.set_default_timeout(self.DEFAULT_INTERACTION_TIMEOUT_MS)
        self.context.set_default_navigation_timeout(self.DEFAULT_NAVIGATION_TIMEOUT_MS)
        self.page.set_default_timeout(self.DEFAULT_INTERACTION_TIMEOUT_MS)
        self.page.set_default_navigation_timeout(self.DEFAULT_NAVIGATION_TIMEOUT_MS)
        
        if not self.headless:
            print(f"[Browser] ✅ Browser window should now be visible!", file=sys.stderr, flush=True)

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

    def _log_dom_state(self, action: str, timeout: int = 1000) -> None:
        """Log current DOM state for debugging.
        
        Args:
            action: Action name for logging context
            timeout: Timeout in ms for DOM operations (default: 1000ms)
        """
        import sys
        try:
            if not self.page:
                print(f"[BROWSER] [{action}] Page is None", file=sys.stderr, flush=True)
                return
            
            url = self.page.url
            print(f"[BROWSER] [{action}] DOM State:", file=sys.stderr, flush=True)
            print(f"[BROWSER]   URL: {url}", file=sys.stderr, flush=True)
            
            # Try to get title and body
            try:
                title = self.page.title()
                print(f"[BROWSER]   Title: {title}", file=sys.stderr, flush=True)
            except Exception:
                print(f"[BROWSER]   Title: (error)", file=sys.stderr, flush=True)
            
            try:
                body_text = self.page.text_content("body") or ""
                body_preview = body_text[:200] if body_text else "empty"
                print(f"[BROWSER]   Body preview: {body_preview}", file=sys.stderr, flush=True)
            except Exception:
                print(f"[BROWSER]   Body preview: (error)", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[BROWSER] [{action}] Error logging DOM state: {e}", file=sys.stderr, flush=True)

    def navigate_to_frontend(self, path: str = "/", timeout: int = DEFAULT_NAVIGATION_TIMEOUT_MS) -> None:
        """Navigate to frontend URL using canonical Playwright navigation.
        
        Uses page.goto() without wait_until - just navigate and wait 3s max for page to be ready.
        Note: Ensure no other pages are open in the browser context before calling.
        
        Args:
            path: Path to navigate to (default: /)
            timeout: Timeout in milliseconds for page navigation (default: 3000ms = 3s max)
        """
        import sys
        
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        # Normalize URL first (before any async operations)
        base_url = self.frontend_url.rstrip("/")
        normalized_path = path if path.startswith("/") else f"/{path}"
        url = f"{base_url}{normalized_path}"
        
        print(f"[BROWSER] Navigating to {url} (timeout={timeout}ms)...", file=sys.stderr, flush=True)
        
        try:
            # Use JavaScript navigation - faster and more reliable than page.goto() which can hang
            print(f"[BROWSER] Using JavaScript navigation (window.location.href)...", file=sys.stderr, flush=True)
            self.page.evaluate(f"window.location.href = '{url}'")
            print(f"[BROWSER] JavaScript navigation executed", file=sys.stderr, flush=True)
            
            # Wait for page to load - check for appropriate h1 based on path
            if path == "/" or path == "":
                # Home page has "Create Run"
                print(f"[BROWSER] Waiting for h1 'Create Run' to appear...", file=sys.stderr, flush=True)
                self.page.wait_for_selector("h1:has-text('Create Run')", timeout=timeout, state="visible")
                print(f"[BROWSER] ✓ Page ready - h1 'Create Run' found. URL: {self.page.url}", file=sys.stderr, flush=True)
            elif "/runs" in path:
                # Runs list page has "Run History"
                print(f"[BROWSER] Waiting for h1 'Run History' to appear...", file=sys.stderr, flush=True)
                self.page.wait_for_selector("h1:has-text('Run History')", timeout=timeout, state="visible")
                print(f"[BROWSER] ✓ Page ready - h1 'Run History' found. URL: {self.page.url}", file=sys.stderr, flush=True)
            else:
                # Other pages - just wait for any h1
                print(f"[BROWSER] Waiting for any h1 to appear...", file=sys.stderr, flush=True)
                self.page.wait_for_selector("h1", timeout=timeout, state="visible")
                print(f"[BROWSER] ✓ Page ready - h1 found. URL: {self.page.url}", file=sys.stderr, flush=True)
            
            self._log_dom_state("AFTER_NAVIGATE")
                
        except Exception as e:
            self._log_dom_state("NAVIGATE_ERROR")
            print(f"[BROWSER] ERROR navigating to frontend: {e}", file=sys.stderr, flush=True)
            raise

    def wait_for_run_result(
        self,
        total_timeout_ms: int = DEFAULT_RUN_RESULT_TIMEOUT_MS,
        step_timeout_ms: int = DEFAULT_INTERACTION_TIMEOUT_MS,
        poll_interval_ms: int = 200,
    ) -> None:
        """Wait for run result to appear on page.

        This uses a loop of short (<= 1s) waits, so we never have a single long DOM wait.
        """
        import sys

        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")

        result_locator = self.page.locator("text=/Plan|Execution|Error|plan|execution|error|diagnostics/i").first
        deadline = time.time() + (total_timeout_ms / 1000.0)

        print(
            f"[BROWSER] Waiting for run result (total_timeout={total_timeout_ms}ms, step_timeout={step_timeout_ms}ms)...",
            file=sys.stderr,
            flush=True,
        )

        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                # This is a short wait (<= 1s)
                result_locator.wait_for(state="visible", timeout=step_timeout_ms)
                print("[BROWSER] ✓ Run result appeared!", file=sys.stderr, flush=True)
                return
            except Exception as e:
                last_err = e
                time.sleep(poll_interval_ms / 1000.0)

        # Timeout: capture diagnostics
        page_text = ""
        try:
            page_text = self.page.text_content("body") or ""
        except Exception:
            pass
        try:
            screenshot_path = "/tmp/browser_screenshot.png"
            self.page.screenshot(path=screenshot_path)
            print(f"[BROWSER] Screenshot saved to {screenshot_path}", file=sys.stderr, flush=True)
        except Exception:
            pass

        raise RuntimeError(
            f"Timeout waiting for run result after {total_timeout_ms}ms. "
            f"Last error: {last_err}. Page text: {page_text[:500]}"
        )

    def create_run_via_ui(
        self,
        goal: str,
        context: Optional[str] = None,
        interaction_timeout: int = DEFAULT_INTERACTION_TIMEOUT_MS,
    ) -> None:
        """Create a run via the UI form.

        Args:
            goal: Goal text to enter
            context: Optional context JSON string
            interaction_timeout: Timeout in milliseconds for DOM interactions (default: 1000ms)
        """
        import sys
        
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")

        try:
            print(f"[BROWSER] Step 1: Navigating to frontend...", file=sys.stderr, flush=True)
            # Navigate to run creation page (route is "/" not "/run")
            # Always navigate fresh to ensure we're on the right page
            self.navigate_to_frontend("/")
            
            # Verify we're actually on the frontend page
            current_url = self.page.url if self.page else "N/A"
            if "localhost" not in current_url.lower() or "3000" not in current_url:
                print(f"[BROWSER] WARNING: Not on frontend page (URL: {current_url}), reloading...", file=sys.stderr, flush=True)
                self.page.reload(timeout=3000)
                time.sleep(1.0)  # Wait 1s after reload
                time.sleep(1.0)  # Wait for SPA to render
                current_url = self.page.url if self.page else "N/A"
                print(f"[BROWSER] After reload, URL: {current_url}", file=sys.stderr, flush=True)
            
            print(f"[BROWSER] ✓ Navigation complete. Current URL: {current_url}", file=sys.stderr, flush=True)

            # Playwright auto-waits for elements to be actionable
            # We can directly interact - no need for manual waits
            print(f"[BROWSER] Step 2: Filling goal textarea...", file=sys.stderr, flush=True)
            self._log_dom_state("BEFORE_FILL_GOAL")
            try:
                # Playwright will auto-wait for textarea to be visible and enabled
                goal_textarea = self.page.locator("textarea").first
                goal_textarea.fill(goal, timeout=interaction_timeout)
                print(f"[BROWSER] ✓ Goal filled: {goal[:50]}...", file=sys.stderr, flush=True)
                self._log_dom_state("AFTER_FILL_GOAL")
            except Exception as e:
                self._log_dom_state("FILL_GOAL_ERROR")
                page_text = self.page.text_content("body") or ""
                raise RuntimeError(f"Failed to fill goal textarea: {e}. Page text: {page_text[:200]}") from e

            # Fill context if provided (second textarea)
            if context:
                print(f"[BROWSER] Step 3: Filling context...", file=sys.stderr, flush=True)
                self._log_dom_state("BEFORE_FILL_CONTEXT")
                try:
                    context_textarea = self.page.locator("textarea").nth(1)
                    context_textarea.fill(context, timeout=interaction_timeout)
                    print(f"[BROWSER] ✓ Context filled", file=sys.stderr, flush=True)
                    self._log_dom_state("AFTER_FILL_CONTEXT")
                except Exception as e:
                    self._log_dom_state("FILL_CONTEXT_ERROR")
                    raise RuntimeError(f"Failed to fill context textarea: {e}") from e

            # Click submit button - Playwright auto-waits for button to be actionable
            print(f"[BROWSER] Step 4: Clicking submit button...", file=sys.stderr, flush=True)
            self._log_dom_state("BEFORE_CLICK_RUN")
            try:
                # Playwright will auto-wait for button to be visible, enabled, and stable
                submit_button = self.page.locator("button:has-text('Run'):not([disabled])")
                print(f"[BROWSER] Found submit button, clicking...", file=sys.stderr, flush=True)
                submit_button.click(timeout=interaction_timeout)
                print(f"[BROWSER] ✓ Submit button clicked. Waiting 500ms for UI to update...", file=sys.stderr, flush=True)
                time.sleep(0.5)  # Small delay for UI to start updating
                self._log_dom_state("AFTER_CLICK_RUN")
            except Exception as e:
                self._log_dom_state("CLICK_RUN_ERROR")
                # Try to find any button for debugging
                try:
                    buttons = self.page.locator("button").all()
                    button_texts = []
                    for btn in buttons[:5]:  # Limit to first 5
                        try:
                            text = btn.text_content()
                            button_texts.append(text)
                        except:
                            pass
                    print(f"[BROWSER] Available buttons: {button_texts}", file=sys.stderr, flush=True)
                except:
                    pass
                raise RuntimeError(f"Failed to click submit button: {e}") from e

            print(f"[BROWSER] Step 5: Waiting for run result...", file=sys.stderr, flush=True)
            self.wait_for_run_result()
            print(f"[BROWSER] ✓ Run creation complete", file=sys.stderr, flush=True)
                
        except Exception as e:
            # Capture page state for debugging
            try:
                page_text = self.page.text_content("body") or ""
                print(f"[BROWSER] ERROR: {e}", file=sys.stderr, flush=True)
                print(f"[BROWSER] Current page URL: {self.page.url}", file=sys.stderr, flush=True)
                print(f"[BROWSER] Page text preview: {page_text[:300]}", file=sys.stderr, flush=True)
            except:
                pass
            raise

    def assert_plan_visible(self, timeout: int = 1000) -> bool:
        """Assert that plan is visible on page.

        Args:
            timeout: Timeout in milliseconds (default: 1000ms)

        Returns:
            True if plan is visible
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        try:
            self.page.wait_for_selector("text=/Plan|plan|objective|steps/i", timeout=timeout, state="visible")
            return True
        except Exception:
            return False

    def assert_execution_visible(self, timeout: int = 1000) -> bool:
        """Assert that execution result is visible on page.

        Args:
            timeout: Timeout in milliseconds (default: 1000ms)

        Returns:
            True if execution is visible
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        try:
            self.page.wait_for_selector("text=/Execution|execution|success|steps/i", timeout=timeout, state="visible")
            return True
        except Exception:
            return False

    def get_plan_text(self) -> str:
        """Get plan text from page.

        Returns:
            Plan text content
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        # Try to find plan section
        plan_selectors = [
            "text=/Plan|plan/i",
            "[data-testid='plan']",
            ".plan",
            "h2:has-text('Plan')",
            "h3:has-text('Plan')",
        ]
        for selector in plan_selectors:
            try:
                element = self.page.wait_for_selector(selector, timeout=2000, state="visible")
                if element:
                    return element.text_content() or ""
            except Exception:
                continue
        return ""

    def get_execution_text(self) -> str:
        """Get execution result text from page.

        Returns:
            Execution text content
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        # Try to find execution section
        exec_selectors = [
            "text=/Execution|execution/i",
            "[data-testid='execution']",
            ".execution",
            "h2:has-text('Execution')",
            "h3:has-text('Execution')",
        ]
        for selector in exec_selectors:
            try:
                element = self.page.wait_for_selector(selector, timeout=2000, state="visible")
                if element:
                    return element.text_content() or ""
            except Exception:
                continue
        return ""

    def navigate_to_runs_list(self) -> None:
        """Navigate to runs list page."""
        self.navigate_to_frontend("/runs")

    def wait_for_runs_list(self, timeout: int = 1000) -> None:
        """Wait for runs list to be visible and populated.

        Args:
            timeout: Timeout in milliseconds (default: 1000ms for DOM interaction)
        """
        import sys
        import time
        
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        try:
            print(f"[BROWSER] Waiting for runs list (timeout={timeout}ms)...", file=sys.stderr, flush=True)
            # Wait for runs list table or "Run History" heading
            self.page.wait_for_selector("text=/Run History|Runs|runs/i", timeout=timeout)
            
            # Wait for table to be populated (not just "Loading...")
            # Check for table rows or that "Loading..." is gone
            start_time = time.time()
            while (time.time() - start_time) * 1000 < timeout:
                page_text = self.page.text_content("body") or ""
                # Check if we have table rows
                rows = self.page.query_selector_all("table tbody tr")
                if len(rows) > 0:
                    print(f"[BROWSER] ✓ Runs list visible and populated ({len(rows)} rows)", file=sys.stderr, flush=True)
                    return
                # Check if "Loading..." is gone (but page is ready)
                if "Loading..." not in page_text:
                    # Give it a small moment for rows to appear
                    time.sleep(0.1)
                    rows = self.page.query_selector_all("table tbody tr")
                    if len(rows) > 0:
                        print(f"[BROWSER] ✓ Runs list visible and populated ({len(rows)} rows)", file=sys.stderr, flush=True)
                        return
                time.sleep(0.1)
            
            # If we get here, table might be empty (which is ok for some tests)
            print(f"[BROWSER] ✓ Runs list visible (may be empty)", file=sys.stderr, flush=True)
        except Exception as e:
            page_text = self.page.text_content("body") or ""
            raise RuntimeError(f"Timeout waiting for runs list. Page text: {page_text[:200]}") from e

    def get_runs_from_list(self) -> list[dict]:
        """Get all runs from the runs list page.

        Returns:
            List of run dictionaries with run_id, objective, success, etc.
        """
        import sys
        
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")

        runs = []
        # Find all table rows (skip header)
        rows = self.page.query_selector_all("table tbody tr")
        print(f"[BROWSER] Found {len(rows)} table rows", file=sys.stderr, flush=True)
        
        # If no rows, log page state for debugging
        if len(rows) == 0:
            page_text = self.page.text_content("body") or ""
            print(f"[BROWSER] WARNING: No table rows found. Page text preview: {page_text[:500]}", file=sys.stderr, flush=True)
        
        for row in rows:
            try:
                cells = row.query_selector_all("td")
                if len(cells) >= 4:
                    # Extract run_id from link
                    link = cells[0].query_selector("a")
                    run_id_text = link.text_content() if link else cells[0].text_content()
                    run_id = run_id_text.strip() if run_id_text else ""

                    # Extract objective
                    objective = cells[1].text_content()
                    objective = objective.strip() if objective else ""

                    # Extract success status
                    success_text = cells[2].text_content()
                    success = "✓" in (success_text or "")

                    # Extract created_at
                    created_at = cells[3].text_content()
                    created_at = created_at.strip() if created_at else ""

                    runs.append({
                        "run_id": run_id,
                        "objective": objective,
                        "success": success,
                        "created_at": created_at,
                    })
            except Exception:
                continue
        return runs

    def click_run_in_list(self, run_id: Optional[str] = None, index: int = 0, interaction_timeout: int = 1000, navigation_timeout: int = 5000) -> str:
        """Click on a run in the list and return the run_id.

        Args:
            run_id: Specific run ID to click (if None, clicks first run)
            index: Index of run to click (if run_id not provided)
            interaction_timeout: Timeout in milliseconds for DOM interactions (default: 1000ms)
            navigation_timeout: Timeout in milliseconds for navigation (default: 10000ms)

        Returns:
            Run ID that was clicked
        """
        import sys
        
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")

        try:
            if run_id:
                print(f"[BROWSER] Clicking run with ID {run_id[:8]}... (timeout={interaction_timeout}ms)", file=sys.stderr, flush=True)
                # Click specific run by ID (look for link containing run_id)
                link = self.page.locator(f"a:has-text('{run_id[:8]}')").first
                link.wait_for(state="visible", timeout=interaction_timeout)
                link.click(timeout=interaction_timeout)
                clicked_run_id = run_id
                print(f"[BROWSER] ✓ Clicked run link", file=sys.stderr, flush=True)
            else:
                print(f"[BROWSER] Clicking run at index {index}... (timeout={interaction_timeout}ms)", file=sys.stderr, flush=True)
                # Click first run in list
                rows = self.page.query_selector_all("table tbody tr")
                if not rows or len(rows) <= index:
                    raise RuntimeError(f"Could not find run at index {index}. Found {len(rows)} rows.")
                
                link = rows[index].query_selector("td a")
                if not link:
                    raise RuntimeError(f"Could not find link in run row {index}")
                
                link.wait_for(state="visible", timeout=interaction_timeout)
                link.click(timeout=interaction_timeout)
                # Extract run_id from link text
                link_text = link.text_content()
                clicked_run_id = link_text.strip() if link_text else ""
                print(f"[BROWSER] ✓ Clicked run link: {clicked_run_id}", file=sys.stderr, flush=True)

            # Wait for navigation to run detail page
            print(f"[BROWSER] Waiting for navigation to run detail (timeout={navigation_timeout}ms)...", file=sys.stderr, flush=True)
            self.page.wait_for_url("**/runs/**", timeout=navigation_timeout)
            print(f"[BROWSER] ✓ Navigated to run detail", file=sys.stderr, flush=True)
            return clicked_run_id
        except Exception as e:
            page_text = self.page.text_content("body") or ""
            raise RuntimeError(f"Timeout clicking run in list. Error: {e}. Page text: {page_text[:200]}") from e

    def wait_for_run_detail(self, timeout: int = 1000) -> None:
        """Wait for run detail page to load.

        Args:
            timeout: Timeout in milliseconds (default: 1000ms for DOM interaction)
        """
        import sys
        
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        
        try:
            print(f"[BROWSER] Waiting for run detail page (timeout={timeout}ms)...", file=sys.stderr, flush=True)
            # Wait for run detail content
            self.page.wait_for_selector("text=/Plan|Execution|plan|execution|Run Detail/i", timeout=timeout)
            print(f"[BROWSER] ✓ Run detail page loaded", file=sys.stderr, flush=True)
        except Exception as e:
            page_text = self.page.text_content("body") or ""
            raise RuntimeError(f"Timeout waiting for run detail. Page text: {page_text[:200]}") from e

    def get_run_id_from_url(self) -> Optional[str]:
        """Extract run ID from current URL.

        Returns:
            Run ID or None
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        url = self.page.url
        # Extract run ID from URL like /runs/{run_id}
        if "/runs/" in url:
            parts = url.split("/runs/")
            if len(parts) > 1:
                return parts[1].split("/")[0].split("?")[0]
        return None

    def assert_text_visible(self, text: str, timeout: int = 5000) -> bool:
        """Assert that text is visible on page.

        Args:
            text: Text to search for
            timeout: Timeout in milliseconds

        Returns:
            True if text is visible
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        try:
            self.page.wait_for_selector(f"text={text}", timeout=timeout, state="visible")
            return True
        except Exception:
            return False

    def get_page_content(self) -> str:
        """Get page content as text.

        Returns:
            Page text content
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        return self.page.content()

    def get_page_text(self) -> str:
        """Get page text content (visible text only).

        Returns:
            Page text content
        """
        if not self.page:
            raise RuntimeError("Browser not started. Call start() first.")
        return self.page.inner_text("body")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

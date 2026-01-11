"""Storage inspection helpers for E2E tests."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class StorageHelper:
    """Helper for inspecting database in E2E tests."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize storage helper.

        Args:
            db_path: Path to SQLite database (default: forge_agent.db in project root)
        """
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            db_path = project_root / "forge_agent.db"
        self.db_path = db_path

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run from database.

        Args:
            run_id: Run ID

        Returns:
            Run data or None if not found
        """
        if not self.db_path.exists():
            return None

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Get latest run from database.

        Returns:
            Latest run data or None if no runs exist
        """
        if not self.db_path.exists():
            return None

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def get_all_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all runs from database.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of run data
        """
        if not self.db_path.exists():
            return []

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def assert_run_exists(self, run_id: str) -> Dict[str, Any]:
        """Assert that run exists in database.

        Args:
            run_id: Run ID

        Returns:
            Run data

        Raises:
            AssertionError: If run does not exist
        """
        run = self.get_run(run_id)
        assert run is not None, f"Run {run_id} not found in database"
        return run

    def assert_run_persisted(self, run_id: str) -> Dict[str, Any]:
        """Assert that run is persisted with required fields.

        Args:
            run_id: Run ID

        Returns:
            Run data

        Raises:
            AssertionError: If run is not properly persisted
        """
        run = self.assert_run_exists(run_id)
        assert "plan_result" in run or run.get("plan_result"), "Run missing plan_result"
        assert run.get("created_at"), "Run missing created_at"
        return run

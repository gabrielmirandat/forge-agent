"""Storage (SQLite) validation checks."""

import json
import sqlite3
from pathlib import Path

from smoke_test.config import DATABASE_PATH


def check_database_exists() -> bool:
    """Check if database file exists.

    Returns:
        True if database file exists
    """
    if not DATABASE_PATH.exists():
        print(f"[Storage Check] Database file not found: {DATABASE_PATH}")
        return False

    print(f"[Storage Check] ✓ Database file exists: {DATABASE_PATH}")
    return True


def get_latest_run() -> tuple[bool, dict | None]:
    """Get the latest run from database.

    Returns:
        Tuple of (success, run_data)
    """
    if not DATABASE_PATH.exists():
        print("[Storage Check] Database file does not exist")
        return False, None

    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get latest run
        cursor.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()

        if not row:
            print("[Storage Check] No runs found in database")
            conn.close()
            return False, None

        # Convert row to dict
        run_data = {
            "run_id": row["run_id"],
            "plan_id": row["plan_id"],
            "objective": row["objective"],
            "plan_result": json.loads(row["plan_result"]),
            "execution_result": (
                json.loads(row["execution_result"]) if row["execution_result"] else None
            ),
            "created_at": row["created_at"],
            "approval_status": row.get("approval_status", "pending"),
        }

        conn.close()

        print(f"[Storage Check] ✓ Latest run found: {run_data['run_id']}")
        return True, run_data
    except Exception as e:
        print(f"[Storage Check] Failed to get latest run: {e}")
        return False, None


def validate_run_structure(run_data: dict) -> bool:
    """Validate run data structure.

    Args:
        run_data: Run data from database

    Returns:
        True if structure is valid
    """
    required_fields = ["run_id", "plan_id", "objective", "plan_result", "created_at"]
    for field in required_fields:
        if field not in run_data:
            print(f"[Storage Check] Run data missing required field: {field}")
            return False

    # Validate plan_result structure
    plan_result = run_data["plan_result"]
    if "plan" not in plan_result:
        print("[Storage Check] plan_result missing 'plan' field")
        return False

    plan = plan_result["plan"]
    if "plan_id" not in plan or "objective" not in plan or "steps" not in plan:
        print("[Storage Check] plan structure invalid")
        return False

    # execution_result can be None if HITL is enabled
    execution_result = run_data.get("execution_result")
    if execution_result is not None:
        if "plan_id" not in execution_result or "success" not in execution_result:
            print("[Storage Check] execution_result structure invalid")
            return False

    print("[Storage Check] ✓ Run structure valid")
    return True


def count_runs() -> int:
    """Count total runs in database.

    Returns:
        Number of runs
    """
    if not DATABASE_PATH.exists():
        return 0

    try:
        conn = sqlite3.connect(str(DATABASE_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM runs")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"[Storage Check] Failed to count runs: {e}")
        return 0


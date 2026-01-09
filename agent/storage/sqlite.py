"""SQLite storage implementation.

Simple SQLite-based storage with auto-migration on startup.
No ORM - uses aiosqlite directly.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import List

import aiosqlite

from agent.observability import (
    get_logger,
    log_event,
    set_run_id,
    storage_operation_duration_seconds,
    storage_operations_total,
)
from agent.runtime.schema import ExecutionResult, PlanResult
from agent.storage.base import NotFoundError, Storage, StorageError
from agent.storage.models import ApprovalStatus, RunRecord, RunSummary


class SQLiteStorage(Storage):
    """SQLite-based storage implementation.

    Single file database with auto-migration on startup.
    JSON stored as TEXT.
    """

    def __init__(self, db_path: str | Path = "forge_agent.db"):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._initialized = False
        self.logger = get_logger("storage", "storage")

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized and migrated."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Create tables if they don't exist
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    plan_result TEXT NOT NULL,
                    execution_result TEXT,
                    created_at REAL NOT NULL,
                    approval_status TEXT NOT NULL DEFAULT 'pending',
                    approval_reason TEXT,
                    approved_at REAL,
                    approved_by TEXT
                )
            """
            )

            # Migrate existing tables - add new columns if they don't exist
            try:
                await db.execute("ALTER TABLE runs ADD COLUMN approval_status TEXT DEFAULT 'pending'")
            except Exception:
                pass  # Column already exists

            try:
                await db.execute("ALTER TABLE runs ADD COLUMN approval_reason TEXT")
            except Exception:
                pass  # Column already exists

            try:
                await db.execute("ALTER TABLE runs ADD COLUMN approved_at REAL")
            except Exception:
                pass  # Column already exists

            try:
                await db.execute("ALTER TABLE runs ADD COLUMN approved_by TEXT")
            except Exception:
                pass  # Column already exists

            # Create index for listing runs
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_created_at 
                ON runs(created_at DESC)
            """
            )

            await db.commit()

        self._initialized = True

    async def save_plan_result(self, plan_result: PlanResult) -> None:
        """Save a plan result.

        Args:
            plan_result: Plan result to save

        Raises:
            StorageError: If save fails
        """
        # Plan results are saved as part of runs
        # This method exists for interface compliance
        # but we don't store plan results separately
        pass

    async def save_execution_result(self, execution_result: ExecutionResult) -> None:
        """Save an execution result.

        Args:
            execution_result: Execution result to save

        Raises:
            StorageError: If save fails
        """
        # Execution results are saved as part of runs
        # This method exists for interface compliance
        # but we don't store execution results separately
        pass

    async def save_run(
        self,
        plan_result: PlanResult,
        execution_result: ExecutionResult | None = None,
        approval_status: ApprovalStatus = ApprovalStatus.PENDING,
    ) -> str:
        """Save a complete run (plan + optional execution).

        Args:
            plan_result: Plan result
            execution_result: Execution result (None if not executed yet)
            approval_status: Approval status (default: PENDING)

        Returns:
            run_id: Unique identifier for the saved run

        Raises:
            StorageError: If save fails
        """
        await self._ensure_initialized()

        import uuid
        import time

        run_id = str(uuid.uuid4())
        created_at = time.time()
        set_run_id(run_id)

        start_time = time.time()
        log_event(self.logger, "storage.run.saved", run_id=run_id, approval_status=approval_status.value)

        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO runs (run_id, plan_id, objective, plan_result, execution_result, created_at, approval_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        run_id,
                        plan_result.plan.plan_id,
                        plan_result.plan.objective,
                        json.dumps(plan_result.model_dump()),
                        json.dumps(execution_result.model_dump()) if execution_result else None,
                        created_at,
                        approval_status.value,
                    ),
                )
                await db.commit()

            duration = time.time() - start_time
            storage_operations_total.labels(operation="save_run", status="success").inc()
            storage_operation_duration_seconds.labels(operation="save_run").observe(duration)

        except Exception as e:
            duration = time.time() - start_time
            storage_operations_total.labels(operation="save_run", status="error").inc()
            storage_operation_duration_seconds.labels(operation="save_run").observe(duration)

            log_event(
                self.logger,
                "storage.failure",
                level="ERROR",
                operation="save_run",
                run_id=run_id,
                error=str(e),
                duration_ms=duration * 1000,
            )

            raise StorageError(f"Failed to save run: {e}") from e

        return run_id

    async def update_run_approval(
        self,
        run_id: str,
        approval_status: ApprovalStatus,
        approved_by: str,
        reason: str | None = None,
    ) -> None:
        """Update approval status of a run.

        Args:
            run_id: Run identifier
            approval_status: New approval status
            approved_by: Who approved/rejected
            reason: Optional reason

        Raises:
            StorageError: If update fails
            NotFoundError: If run not found
        """
        await self._ensure_initialized()

        import time

        approved_at = time.time()

        start_time = time.time()
        log_event(
            self.logger,
            "storage.run.updated",
            run_id=run_id,
            operation="update_approval",
            approval_status=approval_status.value,
        )

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                # Check if run exists and is PENDING
                async with db.execute(
                    "SELECT approval_status FROM runs WHERE run_id = ?", (run_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row is None:
                        raise NotFoundError(f"Run not found: {run_id}")
                    if row["approval_status"] != ApprovalStatus.PENDING.value:
                        raise StorageError(
                            f"Run {run_id} is not pending (current status: {row['approval_status']})"
                        )

                # Update approval status
                await db.execute(
                    """
                    UPDATE runs
                    SET approval_status = ?, approved_by = ?, approval_reason = ?, approved_at = ?
                    WHERE run_id = ?
                """,
                    (approval_status.value, approved_by, reason, approved_at, run_id),
                )
                await db.commit()

            duration = time.time() - start_time
            storage_operations_total.labels(operation="update_approval", status="success").inc()
            storage_operation_duration_seconds.labels(operation="update_approval").observe(duration)

        except NotFoundError:
            raise
        except Exception as e:
            duration = time.time() - start_time
            storage_operations_total.labels(operation="update_approval", status="error").inc()
            storage_operation_duration_seconds.labels(operation="update_approval").observe(duration)

            log_event(
                self.logger,
                "storage.failure",
                level="ERROR",
                operation="update_approval",
                run_id=run_id,
                error=str(e),
                duration_ms=duration * 1000,
            )

            raise StorageError(f"Failed to update approval: {e}") from e

    async def update_run_execution(
        self, run_id: str, execution_result: ExecutionResult
    ) -> None:
        """Update execution result for an approved run.

        Args:
            run_id: Run identifier
            execution_result: Execution result

        Raises:
            StorageError: If update fails
            NotFoundError: If run not found
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Check if run exists
                async with db.execute(
                    "SELECT run_id FROM runs WHERE run_id = ?", (run_id,)
                ) as cursor:
                    if await cursor.fetchone() is None:
                        raise NotFoundError(f"Run not found: {run_id}")

                # Update execution result
                await db.execute(
                    "UPDATE runs SET execution_result = ? WHERE run_id = ?",
                    (json.dumps(execution_result.model_dump()), run_id),
                )
                await db.commit()
        except NotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to update execution: {e}") from e

    async def get_run(self, run_id: str) -> RunRecord:
        """Get a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            RunRecord with full run data

        Raises:
            StorageError: If retrieval fails
            NotFoundError: If run not found
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row is None:
                        raise NotFoundError(f"Run not found: {run_id}")

                    execution_result = None
                    if row["execution_result"]:
                        execution_result = json.loads(row["execution_result"])

                    # Handle approval fields (may not exist in old records)
                    approval_status_str = row["approval_status"] if "approval_status" in row.keys() else "pending"
                    approval_reason = row["approval_reason"] if "approval_reason" in row.keys() else None
                    approved_at = row["approved_at"] if "approved_at" in row.keys() else None
                    approved_by = row["approved_by"] if "approved_by" in row.keys() else None

                    return RunRecord(
                        run_id=row["run_id"],
                        plan_id=row["plan_id"],
                        objective=row["objective"],
                        plan_result=json.loads(row["plan_result"]),
                        execution_result=execution_result,
                        created_at=row["created_at"],
                        approval_status=ApprovalStatus(approval_status_str),
                        approval_reason=approval_reason,
                        approved_at=approved_at,
                        approved_by=approved_by,
                    )
        except NotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to get run: {e}") from e

    async def list_runs(self, limit: int = 20, offset: int = 0) -> List[RunSummary]:
        """List runs with pagination.

        Args:
            limit: Maximum number of runs to return
            offset: Number of runs to skip

        Returns:
            List of RunSummary objects

        Raises:
            StorageError: If retrieval fails
        """
        await self._ensure_initialized()

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    """
                    SELECT run_id, plan_id, objective, execution_result, created_at, approval_status
                    FROM runs
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                ) as cursor:
                    rows = await cursor.fetchall()
                    summaries = []
                    for row in rows:
                        # Parse execution_result to get success status (if exists)
                        success = False
                        if row["execution_result"]:
                            exec_result = json.loads(row["execution_result"])
                            success = exec_result.get("success", False)
                        # If not executed yet, success is False
                        summaries.append(
                            RunSummary(
                                run_id=row["run_id"],
                                plan_id=row["plan_id"],
                                objective=row["objective"],
                                success=success,
                                created_at=row["created_at"],
                            )
                        )
                    return summaries
        except Exception as e:
            raise StorageError(f"Failed to list runs: {e}") from e


"""Storage abstraction interface.

Storage is passive - it never mutates data or affects execution behavior.
"""

from abc import ABC, abstractmethod

from agent.runtime.schema import ExecutionResult, PlanResult


class Storage(ABC):
    """Abstract storage interface for persisting plans, executions, and runs.

    Storage is passive:
    - No business logic
    - Never mutates data
    - Failures surface explicitly
    - Never affects execution behavior
    """

    @abstractmethod
    async def save_plan_result(self, plan_result: PlanResult) -> None:
        """Save a plan result.

        Args:
            plan_result: Plan result to save

        Raises:
            StorageError: If save fails
        """
        pass

    @abstractmethod
    async def save_execution_result(self, execution_result: ExecutionResult) -> None:
        """Save an execution result.

        Args:
            execution_result: Execution result to save

        Raises:
            StorageError: If save fails
        """
        pass

    @abstractmethod
    async def save_run(
        self,
        plan_result: PlanResult,
        execution_result: ExecutionResult | None = None,
        approval_status: "ApprovalStatus" = None,
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
        pass

    @abstractmethod
    async def update_run_approval(
        self,
        run_id: str,
        approval_status: "ApprovalStatus",
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> "RunRecord":
        """Get a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            RunRecord with full run data

        Raises:
            StorageError: If retrieval fails
            NotFoundError: If run not found
        """
        pass

    @abstractmethod
    async def list_runs(self, limit: int = 20, offset: int = 0) -> list["RunSummary"]:
        """List runs with pagination.

        Args:
            limit: Maximum number of runs to return
            offset: Number of runs to skip

        Returns:
            List of RunSummary objects

        Raises:
            StorageError: If retrieval fails
        """
        pass


class StorageError(Exception):
    """Base exception for storage operations."""

    pass


class NotFoundError(StorageError):
    """Raised when a requested resource is not found."""

    pass


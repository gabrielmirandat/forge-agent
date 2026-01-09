"""Approval endpoints - approve/reject runs."""

import logging
import uuid
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status

from agent.config.loader import AgentConfig
from agent.runtime.executor import Executor
from agent.runtime.schema import ExecutionPolicy
from agent.storage import ApprovalStatus, NotFoundError, Storage, StorageError
from api.dependencies import get_config, get_executor, get_storage, get_tool_registry
from api.schemas.approval import ApproveRequest, ApproveResponse, RejectRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/runs/{run_id}/approve",
    response_model=ApproveResponse,
    status_code=status.HTTP_200_OK,
)
async def approve_run(
    run_id: str,
    request: ApproveRequest,
    storage: Storage = Depends(get_storage),
) -> ApproveResponse:
    """Approve a planned run and trigger execution.

    Args:
        run_id: Run identifier
        request: Approval request with approver and optional reason/policy
        storage: Storage instance (injected)

    Returns:
        ApproveResponse with approval status and execution result

    Raises:
        HTTPException: If approval or execution fails
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    set_run_id(run_id)

    start_time = time.time()
    log_event(
        logger,
        "api.request.started",
        endpoint=f"/runs/{run_id}/approve",
        method="POST",
        run_id=run_id,
    )

    try:
        # Step 1: Get run and validate it's PENDING
        run = await storage.get_run(run_id)

        if run.approval_status != ApprovalStatus.PENDING:
            logger.warning(
                f"[{request_id}] Run {run_id} is not pending (status: {run.approval_status})"
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": f"Run is not pending (current status: {run.approval_status.value})"
                },
            )

        # Step 2: Update approval status
        try:
            # Calculate time-to-approval
            time_to_approval = time.time() - run.created_at
            approval_pending_duration_seconds.observe(time_to_approval)

            await storage.update_run_approval(
                run_id, ApprovalStatus.APPROVED, request.approved_by, request.reason
            )
            approvals_total.labels(status="approved").inc()

            log_event(
                logger,
                "approval.approved",
                run_id=run_id,
                approved_by=request.approved_by,
                time_to_approval_seconds=time_to_approval,
            )
        except StorageError as e:
            logger.error(f"[{request_id}] Failed to update approval: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Failed to update approval"},
            ) from e

        # Step 3: Parse ExecutionPolicy if provided
        execution_policy = None
        if request.execution_policy:
            execution_policy = ExecutionPolicy(**request.execution_policy)

        # Step 4: Execute plan
        from agent.runtime.schema import Plan

        plan = Plan(**run.plan_result["plan"])
        config = get_config()
        tool_registry = get_tool_registry(config)
        executor = Executor(config, tool_registry, execution_policy)

        execution_result = await executor.execute(plan)

        logger.info(
            f"[{request_id}] Execution completed - success: {execution_result.success}"
        )

        # Step 5: Update run with execution result
        try:
            await storage.update_run_execution(run_id, execution_result)
            logger.info(f"[{request_id}] Execution result persisted")
        except StorageError as e:
            logger.error(f"[{request_id}] Failed to persist execution: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Failed to persist execution result"},
            ) from e

        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/runs/{run_id}/approve", method="POST", status="200").inc()
        api_request_duration_seconds.labels(endpoint="/runs/{run_id}/approve").observe(duration)

        log_event(
            logger,
            "api.request.completed",
            endpoint=f"/runs/{run_id}/approve",
            method="POST",
            status=200,
            duration_ms=duration * 1000,
            run_id=run_id,
        )

        return ApproveResponse(
            run_id=run_id,
            approval_status=ApprovalStatus.APPROVED.value,
            execution_result=execution_result.model_dump(),
        )

    except NotFoundError as e:
        logger.warning(f"[{request_id}] Run not found: {run_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Run not found: {run_id}"},
        ) from e

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e


@router.post(
    "/runs/{run_id}/reject",
    status_code=status.HTTP_200_OK,
)
async def reject_run(
    run_id: str,
    request: RejectRequest,
    storage: Storage = Depends(get_storage),
) -> Dict[str, str]:
    """Reject a planned run.

    Execution is never triggered.

    Args:
        run_id: Run identifier
        request: Rejection request with rejector and reason
        storage: Storage instance (injected)

    Returns:
        Success message

    Raises:
        HTTPException: If rejection fails
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)
    set_run_id(run_id)

    start_time = time.time()
    log_event(
        logger,
        "api.request.started",
        endpoint=f"/runs/{run_id}/reject",
        method="POST",
        run_id=run_id,
    )

    try:
        # Step 1: Get run and validate it's PENDING
        run = await storage.get_run(run_id)

        if run.approval_status != ApprovalStatus.PENDING:
            logger.warning(
                f"[{request_id}] Run {run_id} is not pending (status: {run.approval_status})"
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error": f"Run is not pending (current status: {run.approval_status.value})"
                },
            )

        # Step 2: Update approval status to REJECTED
        try:
            await storage.update_run_approval(
                run_id, ApprovalStatus.REJECTED, request.rejected_by, request.reason
            )
            approvals_total.labels(status="rejected").inc()

            log_event(
                logger,
                "approval.rejected",
                run_id=run_id,
                rejected_by=request.rejected_by,
                reason=request.reason,
            )
        except StorageError as e:
            logger.error(f"[{request_id}] Failed to update rejection: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "Failed to update rejection"},
            ) from e

        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/runs/{run_id}/reject", method="POST", status="200").inc()
        api_request_duration_seconds.labels(endpoint="/runs/{run_id}/reject").observe(duration)

        log_event(
            logger,
            "api.request.completed",
            endpoint=f"/runs/{run_id}/reject",
            method="POST",
            status=200,
            duration_ms=duration * 1000,
            run_id=run_id,
        )

        return {"status": "rejected", "run_id": run_id}

    except NotFoundError as e:
        logger.warning(f"[{request_id}] Run not found: {run_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Run not found: {run_id}"},
        ) from e

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e


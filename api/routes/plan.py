"""Planning endpoint - calls Planner only."""

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from agent.observability import (
    get_logger,
    log_event,
    planner_duration_seconds,
    planner_requests_total,
    planner_validation_errors_total,
    set_request_id,
    trace_span,
)
from agent.runtime.planner import Planner
from agent.runtime.schema import (
    InvalidPlanError,
    LLMCommunicationError,
    PlanningError,
)
from api.dependencies import get_planner
from api.schemas.plan import PlanRequest, PlanResponse

router = APIRouter()
logger = get_logger("api.plan", "api")


@router.post("/plan", response_model=PlanResponse, status_code=status.HTTP_200_OK)
async def plan(
    request: PlanRequest,
    planner: Planner = Depends(get_planner),
) -> PlanResponse:
    """Generate execution plan from goal.

    Calls Planner only. Does NOT execute anything.

    Args:
        request: Plan request with goal and optional context
        planner: Planner instance (injected)

    Returns:
        PlanResponse with plan and diagnostics

    Raises:
        HTTPException: If planning fails
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    start_time = time.time()
    log_event(logger, "api.request.started", endpoint="/plan", method="POST")

    try:
        with trace_span("plan", attributes={"goal": request.goal[:500]}):  # Increased from 100 to 500 chars
            # Direct passthrough to Planner.plan()
            plan_result = await planner.plan(request.goal, request.context)

            # Map internal PlanResult to API response
            response = PlanResponse(
                plan=plan_result.plan.model_dump(),
                diagnostics=plan_result.diagnostics.model_dump(),
            )

            duration = time.time() - start_time
            planner_duration_seconds.observe(duration)
            planner_requests_total.labels(status="success").inc()

            log_event(
                logger,
                "api.request.completed",
                endpoint="/plan",
                method="POST",
                status=200,
                duration_ms=duration * 1000,
                plan_id=plan_result.plan.plan_id,
            )

            return response

    except InvalidPlanError as e:
        duration = time.time() - start_time
        planner_requests_total.labels(status="validation_error").inc()
        planner_validation_errors_total.inc()
        api_requests_total.labels(endpoint="/plan", method="POST", status="422").inc()
        api_request_duration_seconds.labels(endpoint="/plan").observe(duration)

        log_event(
            logger,
            "api.request.failed",
            level="ERROR",
            endpoint="/plan",
            method="POST",
            status=422,
            duration_ms=duration * 1000,
            error=str(e),
        )

        # Include diagnostics if available
        error_detail = {"error": str(e)}
        if hasattr(e, "diagnostics") and e.diagnostics:
            error_detail["diagnostics"] = e.diagnostics.model_dump()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_detail,
        ) from e

    except LLMCommunicationError as e:
        duration = time.time() - start_time
        planner_requests_total.labels(status="llm_error").inc()
        api_requests_total.labels(endpoint="/plan", method="POST", status="502").inc()
        api_request_duration_seconds.labels(endpoint="/plan").observe(duration)

        log_event(
            logger,
            "api.request.failed",
            level="ERROR",
            endpoint="/plan",
            method="POST",
            status=502,
            duration_ms=duration * 1000,
            error=str(e),
        )

        error_detail = {"error": str(e)}
        if hasattr(e, "diagnostics") and e.diagnostics:
            error_detail["diagnostics"] = e.diagnostics.model_dump()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error_detail,
        ) from e

    except PlanningError as e:
        duration = time.time() - start_time
        planner_requests_total.labels(status="error").inc()
        api_requests_total.labels(endpoint="/plan", method="POST", status="500").inc()
        api_request_duration_seconds.labels(endpoint="/plan").observe(duration)

        log_event(
            logger,
            "api.request.failed",
            level="ERROR",
            endpoint="/plan",
            method="POST",
            status=500,
            duration_ms=duration * 1000,
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Planning failed"},
        ) from e

    except Exception as e:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/plan", method="POST", status="500").inc()
        api_request_duration_seconds.labels(endpoint="/plan").observe(duration)

        log_event(
            logger,
            "api.request.failed",
            level="ERROR",
            endpoint="/plan",
            method="POST",
            status=500,
            duration_ms=duration * 1000,
            error=str(e),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e


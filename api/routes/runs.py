"""Run history endpoints - read-only."""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from agent.storage import NotFoundError, Storage, StorageError
from api.dependencies import get_storage
from api.schemas.runs import RunDetailResponse, RunsListResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/runs", response_model=RunsListResponse, status_code=status.HTTP_200_OK)
async def list_runs(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of runs to return"),
    offset: int = Query(default=0, ge=0, description="Number of runs to skip"),
    storage: Storage = Depends(get_storage),
) -> RunsListResponse:
    """List runs with pagination.

    Args:
        limit: Maximum number of runs to return (1-100)
        offset: Number of runs to skip
        storage: Storage instance (injected)

    Returns:
        RunsListResponse with list of run summaries

    Raises:
        HTTPException: If retrieval fails
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] GET /runs - limit={limit}, offset={offset}")

    try:
        runs = await storage.list_runs(limit=limit, offset=offset)

        logger.info(f"[{request_id}] Retrieved {len(runs)} runs")
        return RunsListResponse(runs=runs, limit=limit, offset=offset)

    except StorageError as e:
        logger.error(f"[{request_id}] Storage error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve runs"},
        ) from e

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e


@router.get(
    "/runs/{run_id}", response_model=RunDetailResponse, status_code=status.HTTP_200_OK
)
async def get_run(
    run_id: str,
    storage: Storage = Depends(get_storage),
) -> RunDetailResponse:
    """Get a run by ID.

    Args:
        run_id: Run identifier
        storage: Storage instance (injected)

    Returns:
        RunDetailResponse with complete run record

    Raises:
        HTTPException: If retrieval fails or run not found
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] GET /runs/{run_id}")

    try:
        run = await storage.get_run(run_id)

        logger.info(f"[{request_id}] Retrieved run {run_id}")
        return RunDetailResponse(run=run)

    except NotFoundError as e:
        logger.warning(f"[{request_id}] Run not found: {run_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Run not found: {run_id}"},
        ) from e

    except StorageError as e:
        logger.error(f"[{request_id}] Storage error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to retrieve run"},
        ) from e

    except Exception as e:
        logger.exception(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error"},
        ) from e


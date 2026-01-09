"""Health check endpoint."""

import time

from fastapi import APIRouter

router = APIRouter()

# Track startup time at module level to avoid circular import
_start_time = time.time()


@router.get("/health")
async def health():
    """Basic liveness check with uptime.

    No dependencies on LLM or tools.
    """
    uptime_seconds = time.time() - _start_time
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": uptime_seconds,
    }


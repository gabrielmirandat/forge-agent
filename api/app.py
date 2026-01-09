"""FastAPI main application."""

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from api.routes import approval, execute, health, plan, run, runs

app = FastAPI(
    title="Forge Agent API",
    description="API for self-hosted autonomous code agent",
    version="0.1.0",
)

# Register routes
app.include_router(health.router, tags=["health"])
app.include_router(plan.router, prefix="/api/v1", tags=["planning"])
app.include_router(execute.router, prefix="/api/v1", tags=["execution"])
app.include_router(run.router, prefix="/api/v1", tags=["orchestration"])
app.include_router(runs.router, prefix="/api/v1", tags=["history"])
app.include_router(approval.router, prefix="/api/v1", tags=["approval"])

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Forge Agent API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "plan": "/api/v1/plan",
            "execute": "/api/v1/execute",
            "run": "/api/v1/run",
        },
    }


"""FastAPI main application."""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from prometheus_client import make_asgi_app

# Load environment variables from .env file
# Look for .env in the project root (parent of api/)
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
else:
    print(f"No .env file found at {env_path} (using system environment variables)")

from api.routes import execute, health, observability, plan, session

app = FastAPI(
    title="Forge Agent API",
    description="API for self-hosted autonomous code agent",
    version="0.1.0",
)

# Register routes
app.include_router(health.router, tags=["health"])
app.include_router(plan.router, prefix="/api/v1", tags=["planning"])
app.include_router(execute.router, prefix="/api/v1", tags=["execution"])
app.include_router(session.router, prefix="/api/v1", tags=["sessions"])
app.include_router(observability.router, prefix="/api/v1", tags=["observability"])

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
            "sessions": "/api/v1/sessions",
        },
    }


"""FastAPI main application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import agent

app = FastAPI(
    title="Forge Agent API",
    description="API for self-hosted autonomous code agent",
    version="0.1.0",
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(agent.router, prefix="/api/v1", tags=["agent"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Forge Agent API", "version": "0.1.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


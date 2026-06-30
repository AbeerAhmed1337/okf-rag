"""
main.py — FastAPI application factory with lifespan, CORS, and router registration.

Entry point:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.exceptions import register_exception_handlers
from app.logger import get_logger
from app.routes import chat, upload
from app.services.neo4j_client import neo4j_client

log = get_logger(__name__)
settings = get_settings()


# ── Lifespan Context Manager ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application-level resources:
      - Neo4j async driver: opened on startup, closed on shutdown.
      - Any other connection pools (Redis, etc.) should be added here.

    This replaces the deprecated @app.on_event("startup") pattern.
    """
    log.info("Starting %s v%s …", settings.APP_NAME, settings.APP_VERSION)

    # ── Startup ────────────────────────────────────────────────────────────────
    await neo4j_client.connect()
    app.state.neo4j = neo4j_client  # expose to request handlers via request.app.state

    log.info("Application startup complete. Listening for requests.")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    log.info("Shutting down %s …", settings.APP_NAME)
    await neo4j_client.close()
    log.info("Shutdown complete.")


# ── Application Factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "A production-grade GraphRAG backend that ingests PDF documents, "
            "compiles them into Open Knowledge Format (OKF) nodes, persists "
            "them to a Neo4j graph database, and answers questions via a "
            "hybrid vector + graph retrieval pipeline powered by DeepSeek V4 Pro."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ───────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception Handlers ─────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ────────────────────────────────────────────────────────────────
    API_PREFIX = "/api/v1"
    app.include_router(upload.router, prefix=API_PREFIX, tags=["Ingest"])
    app.include_router(chat.router,   prefix=API_PREFIX, tags=["Chat"])

    # ── Health & Root Endpoints ────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "service": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "docs": "/docs",
            }
        )

    @app.get("/health", tags=["Health"], summary="Liveness probe")
    async def health() -> JSONResponse:
        """Returns 200 OK when the application is running."""
        return JSONResponse(content={"status": "healthy", "version": settings.APP_VERSION})

    return app


# ── WSGI/ASGI entry point ──────────────────────────────────────────────────────
app = create_app()

"""
exceptions.py — Domain-specific exception hierarchy and FastAPI exception handlers.
Centralising error types keeps route handlers thin and uniform.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.logger import get_logger

log = get_logger(__name__)


# ── Domain Exceptions ─────────────────────────────────────────────────────────

class OKFBaseError(Exception):
    """Root exception for all OKF-RAG errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_message: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.default_message
        super().__init__(self.detail)


class PDFParseError(OKFBaseError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    default_message = "Failed to parse the supplied PDF file."


class OKFCompilationError(OKFBaseError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_message = "Multi-agent OKF compilation pipeline failed."


class Neo4jPersistenceError(OKFBaseError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_message = "Could not persist data to the graph database."


class EmbeddingError(OKFBaseError):
    status_code = status.HTTP_502_BAD_GATEWAY
    default_message = "Embedding generation failed."


class ChatStreamError(OKFBaseError):
    status_code = status.HTTP_502_BAD_GATEWAY
    default_message = "LLM streaming failed."


class NotFoundError(OKFBaseError):
    status_code = status.HTTP_404_NOT_FOUND
    default_message = "The requested resource was not found."


class FileTooLargeError(OKFBaseError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    default_message = "Uploaded file exceeds the maximum allowed size."


# ── FastAPI Exception Handlers ─────────────────────────────────────────────────

async def okf_exception_handler(request: Request, exc: OKFBaseError) -> JSONResponse:
    log.error(
        "OKF domain error: %s | path=%s | detail=%s",
        type(exc).__name__,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "detail": exc.detail,
            "path": str(request.url.path),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception at %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An unexpected internal error occurred.",
            "path": str(request.url.path),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI application instance."""
    app.add_exception_handler(OKFBaseError, okf_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)

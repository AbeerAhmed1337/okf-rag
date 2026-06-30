"""
services/neo4j_client.py — Async Neo4j driver lifecycle and session factory.

The driver is initialised once during application startup (via the lifespan
context manager in main.py) and is stored as application state so that every
request can borrow a lightweight session without the overhead of reconnecting.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from app.config import get_settings
from app.exceptions import Neo4jPersistenceError
from app.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


class Neo4jClient:
    """
    Thin wrapper around the Neo4j async driver.

    Responsibilities:
    - Open and close the driver (called from lifespan hooks).
    - Provide an ``async with`` session context manager for callers.
    - Surface domain-level errors via Neo4jPersistenceError.
    """

    def __init__(self) -> None:
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialise the driver and verify connectivity."""
        log.info("Connecting to Neo4j at %s …", settings.NEO4J_URI)
        self._driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            max_connection_pool_size=50,
        )
        try:
            await self._driver.verify_connectivity()
            log.info("Neo4j connection verified ✓")
        except Exception as exc:
            log.error("Neo4j connectivity check failed: %s", exc)
            raise Neo4jPersistenceError(
                f"Cannot reach Neo4j at {settings.NEO4J_URI}: {exc}"
            ) from exc

    async def close(self) -> None:
        """Gracefully close all pooled connections."""
        if self._driver:
            await self._driver.close()
            log.info("Neo4j driver closed.")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Yield a database session; propagate errors as Neo4jPersistenceError."""
        if self._driver is None:
            raise Neo4jPersistenceError("Neo4j driver is not initialised.")
        async with self._driver.session(database=settings.NEO4J_DATABASE) as sess:
            try:
                yield sess
            except Exception as exc:
                log.error("Neo4j session error: %s", exc)
                raise Neo4jPersistenceError(str(exc)) from exc


# ── Singleton instance (shared across the app via request.app.state) ───────────
neo4j_client = Neo4jClient()

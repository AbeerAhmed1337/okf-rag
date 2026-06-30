"""
logger.py — Structured, levelled logging configuration.
Uses Python's stdlib logging with a JSON-friendly formatter for production.
"""

import logging
import sys
from app.config import get_settings

settings = get_settings()


def _build_formatter() -> logging.Formatter:
    """Return a compact formatter; swap for structlog/JSON in production."""
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    return logging.Formatter(fmt=fmt, datefmt=datefmt)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger attached to stdout with the configured level.

    Usage:
        from app.logger import get_logger
        log = get_logger(__name__)
        log.info("service started")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_build_formatter())
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False
    return logger

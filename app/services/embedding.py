"""
services/embedding.py — Async vector embedding generation.

Uses the OpenAI SDK pointed at the configured embedding model endpoint.
Swap the base_url to use Azure OpenAI, Cohere, or a local sentence-transformers
service without changing any caller code.
"""

from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from app.config import get_settings
from app.exceptions import EmbeddingError
from app.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


def _get_client() -> AsyncOpenAI:
    """Return a lazily-constructed async OpenAI client."""
    return AsyncOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )


async def generate_embedding(text: str) -> list[float]:
    """
    Generate a dense vector embedding for a single text string.

    Parameters
    ----------
    text : str
        Input text to embed (will be truncated by the API if > token limit).

    Returns
    -------
    list[float]
        A float vector of length ``settings.EMBEDDING_DIMENSIONS``.

    Raises
    ------
    EmbeddingError
        If the API call fails for any reason.

    Production note
    ---------------
    For bulk embeddings (e.g. at upload time), use ``generate_embeddings_batch``
    below to amortise API round-trips.
    """
    log.debug("Generating embedding for text (len=%d) …", len(text))
    try:
        client = _get_client()
        response = await client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text,
        )
        vector: list[float] = response.data[0].embedding
        log.debug("Embedding generated. Dimensions: %d", len(vector))
        return vector
    except Exception as exc:
        log.error("Embedding API error: %s", exc)
        raise EmbeddingError(f"Embedding generation failed: {exc}") from exc


async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts concurrently.

    Batches are processed via asyncio.gather to minimise wall-clock time.
    In production, consider chunking into groups of ~100 to respect rate limits.
    """
    if not texts:
        return []

    log.info("Generating %d embeddings in batch …", len(texts))
    try:
        tasks = [generate_embedding(t) for t in texts]
        results: list[list[float]] = await asyncio.gather(*tasks)
        return results
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError(f"Batch embedding failed: {exc}") from exc

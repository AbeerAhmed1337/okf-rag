"""
services/embedding.py — Local vector embedding using sentence-transformers.

Runs entirely on-device — no API key, no network call, no cost.

Model: all-MiniLM-L6-v2
  - 384-dimensional output vectors
  - Trained on 1B+ sentence pairs for semantic similarity
  - ~90 MB download on first run (cached automatically in ~/.cache/torch/)
  - CPU inference: ~5–15 ms per sentence

The model is loaded once at module import time (singleton pattern) to avoid
the ~2–3 second cold-start cost on every request.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.exceptions import EmbeddingError
from app.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Load and cache the sentence-transformers model.

    Called once on first use; subsequent calls return the cached instance.
    The model name is read from settings so it is configurable via .env.
    """
    model_name = settings.EMBEDDING_MODEL
    log.info("Loading local embedding model: %s …", model_name)
    model = SentenceTransformer(model_name)
    log.info("Embedding model loaded. Output dimensions: %d", model.get_sentence_embedding_dimension())
    return model


async def generate_embedding(text: str) -> list[float]:
    """
    Generate a dense vector embedding for a single text string.

    Runs the CPU-bound SentenceTransformer inference in a thread-pool
    executor so the FastAPI event loop is never blocked.

    Parameters
    ----------
    text : str
        Input text to embed (truncated to model's max token limit ~256).

    Returns
    -------
    list[float]
        Float vector of length ``settings.EMBEDDING_DIMENSIONS`` (384).

    Raises
    ------
    EmbeddingError
        If encoding fails for any reason.
    """
    try:
        loop = asyncio.get_event_loop()
        model = _get_model()

        # Run synchronous .encode() in thread pool — keeps async event loop free
        vector: list[float] = await loop.run_in_executor(
            None,
            lambda: model.encode(text, normalize_embeddings=True).tolist(),
        )

        log.debug("Embedding generated. dim=%d text_len=%d", len(vector), len(text))
        return vector

    except Exception as exc:
        log.error("Local embedding error: %s", exc)
        raise EmbeddingError(f"Local embedding generation failed: {exc}") from exc


async def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts in a single batched call.

    sentence-transformers' .encode() is already optimised for batches
    (handles padding + batching internally), so we call it once with all
    texts rather than N individual calls.

    Parameters
    ----------
    texts : list[str]
        List of text strings to embed.

    Returns
    -------
    list[list[float]]
        List of float vectors in the same order as ``texts``.
    """
    if not texts:
        return []

    log.info("Batch embedding %d texts locally …", len(texts))

    try:
        loop = asyncio.get_event_loop()
        model = _get_model()

        vectors: list[list[float]] = await loop.run_in_executor(
            None,
            lambda: model.encode(texts, normalize_embeddings=True, batch_size=32).tolist(),
        )

        log.info("Batch embedding complete. count=%d dim=%d", len(vectors), len(vectors[0]))
        return vectors

    except Exception as exc:
        log.error("Batch embedding error: %s", exc)
        raise EmbeddingError(f"Local batch embedding failed: {exc}") from exc

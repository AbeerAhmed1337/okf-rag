"""
config.py — Application configuration via Pydantic BaseSettings.
All secrets and environment variables are loaded from a .env file at startup.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized, type-safe application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    APP_NAME: str = "OKF PDF-RAG Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"

    # ── DeepSeek / OpenAI-compatible LLM ──────────────────────────────────────
    DEEPSEEK_API_KEY: str = "sk-placeholder"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"  # Replaces deprecated deepseek-reasoner (2026-07-24)

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local sentence-transformers model, 384-dim
    EMBEDDING_DIMENSIONS: int = 384

    # ── MongoDB / GridFS ──────────────────────────────────────────────────────
    MONGO_URI: str = ""          # e.g. mongodb://localhost:27017  (empty = disabled)
    MONGO_DB_NAME: str = "okf_rag"

    # ── File Upload ───────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "/tmp/okf_uploads"
    MAX_UPLOAD_SIZE_MB: int = 50


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()

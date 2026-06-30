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
    DEEPSEEK_MODEL: str = "deepseek-reasoner"

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # ── File Upload ───────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "/tmp/okf_uploads"
    MAX_UPLOAD_SIZE_MB: int = 50


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()

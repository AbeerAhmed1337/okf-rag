"""
schemas/upload.py — Request & response models for the /upload endpoint.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class UploadStatus(str, Enum):
    ACCEPTED = "accepted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class OKFNodeMeta(BaseModel):
    """Metadata for a single OKF node persisted to Neo4j."""

    node_id: str = Field(..., description="Neo4j internal node identifier")
    label: str = Field(..., description="OKF node label, e.g. Concept, Block")
    title: str
    okf_type: str = Field(..., description="OKF semantic type")
    yaml_valid: bool = Field(..., description="Whether YAML front-matter passed validation")


class UploadResponse(BaseModel):
    """
    202 Accepted payload returned after a successful PDF ingest.
    The task_id can be used to poll status (future enhancement).
    """

    task_id: UUID
    status: UploadStatus = UploadStatus.ACCEPTED
    filename: str
    page_count: int | None = None
    okf_nodes: list[OKFNodeMeta] = Field(default_factory=list)
    concepts_extracted: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    message: str = "PDF accepted. OKF compilation pipeline initiated."

    model_config = {"json_schema_extra": {
        "example": {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "accepted",
            "filename": "knowledge_base.pdf",
            "page_count": 42,
            "okf_nodes": [
                {
                    "node_id": "4:abc123:1",
                    "label": "OKFBlock",
                    "title": "Introduction to Graph RAG",
                    "okf_type": "concept",
                    "yaml_valid": True,
                }
            ],
            "concepts_extracted": 17,
            "created_at": "2026-06-30T12:00:00Z",
            "message": "PDF accepted. OKF compilation pipeline initiated.",
        }
    }}

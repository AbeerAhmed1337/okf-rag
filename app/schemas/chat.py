"""
schemas/chat.py — Request & response models for the /chat endpoint.
"""

from __future__ import annotations

from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Payload for a single chat turn."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="User's natural-language query.",
    )
    conversation_id: UUID | None = Field(
        default=None,
        description="Existing conversation UUID for multi-turn sessions. "
                    "Omit to start a new conversation.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Explain the concept of vector embeddings in GraphRAG.",
                "conversation_id": None,
            }
        }
    }


class ChatStreamEvent(BaseModel):
    """
    Single Server-Sent Event (SSE) frame schema.
    The client should inspect `event_type` to route tokens correctly.
    """

    event_type: str = Field(
        ...,
        description="One of: 'thought' | 'answer' | 'context' | 'done' | 'error'",
    )
    content: str = Field(..., description="Token or message payload.")
    conversation_id: UUID = Field(default_factory=uuid4)

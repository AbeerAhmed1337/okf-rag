"""
routes/chat.py — POST /api/v1/chat

Chat pipeline:
  1. Validate the incoming ChatRequest payload.
  2. Generate a vector embedding for the user's query.
  3. Run hybrid vector + graph retrieval against Neo4j.
  4. Stream the DeepSeek V4 Pro response (with thought/answer separation)
     back to the client as SSE.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request, status
from fastapi.responses import StreamingResponse

from app.exceptions import EmbeddingError
from app.logger import get_logger
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_stream import stream_chat_response, generate_chat_response
from app.services.embedding import generate_embedding
from app.services.graph_store import query_graph_context
from app.services.neo4j_client import neo4j_client

log = get_logger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    summary="Get a GraphRAG-grounded chat response from DeepSeek V4 Pro",
    response_description="JSON response with answer and thought",
    status_code=status.HTTP_200_OK,
    response_model=ChatResponse,
    responses={
        400: {"description": "Invalid request payload"},
        502: {"description": "Embedding or LLM API failure"},
    },
)
async def chat(
    request: Request,
    body: ChatRequest,
) -> ChatResponse:
    """
    **Chat Endpoint**

    Accepts a user message (and optional conversation_id for multi-turn
    history) and returns a JSON response containing the final answer and
    internal reasoning.
    """
    log.info(
        "Chat request received. conversation_id=%s message_len=%d",
        body.conversation_id,
        len(body.message),
    )

    # ── 1. Embed the user query ────────────────────────────────────────────────
    try:
        query_embedding = await generate_embedding(body.message)
    except EmbeddingError as exc:
        log.error("Embedding failed for chat query: %s", exc.detail)
        raise

    # ── 2. Hybrid graph retrieval ──────────────────────────────────────────────
    async with neo4j_client.session() as session:
        context_records = await query_graph_context(
            session=session,
            query_embedding=query_embedding,
            top_k=5,
        )

    # ── 3. Generate LLM response (non-streaming) ───────────────────────────────
    conversation_id: UUID | None = body.conversation_id

    response_data = await generate_chat_response(
        user_message=body.message,
        context_records=context_records,
        conversation_id=conversation_id,
    )

    return ChatResponse(**response_data)

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
from app.schemas.chat import ChatRequest
from app.services.chat_stream import stream_chat_response
from app.services.embedding import generate_embedding
from app.services.graph_store import query_graph_context
from app.services.neo4j_client import neo4j_client

log = get_logger(__name__)
router = APIRouter()


@router.post(
    "/chat",
    summary="Stream a GraphRAG-grounded chat response from DeepSeek V4 Pro",
    response_description="SSE stream of typed token events",
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "text/event-stream of SSE frames",
            "content": {
                "text/event-stream": {
                    "example": (
                        'data: {"event_type":"context","content":"Retrieved 3 blocks","conversation_id":"..."}\n\n'
                        'data: {"event_type":"thought","content":"Let me think…","conversation_id":"..."}\n\n'
                        'data: {"event_type":"answer","content":"Graph RAG combines…","conversation_id":"..."}\n\n'
                        'data: {"event_type":"done","content":"","conversation_id":"..."}\n\n'
                    )
                }
            },
        },
        400: {"description": "Invalid request payload"},
        502: {"description": "Embedding or LLM API failure"},
    },
)
async def chat(
    request: Request,
    body: ChatRequest,
) -> StreamingResponse:
    """
    **Streaming Chat Endpoint**

    Accepts a user message (and optional conversation_id for multi-turn
    history) and returns a Server-Sent Events stream.

    **SSE event types:**
    | Type      | Description                                        |
    |-----------|----------------------------------------------------|
    | `context` | Summary of retrieved OKF context blocks            |
    | `thought` | DeepSeek internal reasoning tokens (`<thought>`)   |
    | `answer`  | Final answer tokens                                |
    | `done`    | End-of-stream signal                               |
    | `error`   | Error message if the stream fails mid-way          |
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

    # ── 3. Stream LLM response ─────────────────────────────────────────────────
    conversation_id: UUID | None = body.conversation_id

    return StreamingResponse(
        content=stream_chat_response(
            user_message=body.message,
            context_records=context_records,
            conversation_id=conversation_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # Disable nginx buffering
            "Connection": "keep-alive",
        },
    )

"""
services/chat_stream.py — DeepSeek V4 Pro streaming chat with LiveThoughtCallbackHandler.

Stream Architecture
-------------------
DeepSeek's reasoning model exposes two token streams inside a single SSE feed:

  • reasoning_content  → internal "thinking" tokens  → emitted as <thought> events
  • content            → final answer tokens          → emitted as <answer> events

The LiveThoughtCallbackHandler intercepts both channels and yields typed SSE
frames that the route handler pushes directly to the HTTP response stream.

SSE Frame Format (text/event-stream)
-------------------------------------
    data: {"event_type": "thought",  "content": "…token…", "conversation_id": "…"}
    data: {"event_type": "answer",   "content": "…token…", "conversation_id": "…"}
    data: {"event_type": "context",  "content": "…yaml…",  "conversation_id": "…"}
    data: {"event_type": "done",     "content": "",         "conversation_id": "…"}
    data: {"event_type": "error",    "content": "…msg…",   "conversation_id": "…"}
"""

from __future__ import annotations

import json
from typing import AsyncGenerator
from uuid import UUID, uuid4

from openai import AsyncOpenAI

from app.config import get_settings
from app.exceptions import ChatStreamError
from app.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


# ── System Prompt Builder ──────────────────────────────────────────────────────

def _build_system_prompt(context_records: list[dict]) -> str:
    """
    Inject retrieved OKF context into the system message.

    Parameters
    ----------
    context_records : list[dict]
        Output from graph_store.query_graph_context().
    """
    if not context_records:
        return (
            "You are an expert knowledge assistant powered by a GraphRAG system. "
            "Answer the user's question as helpfully as possible. "
            "If you lack sufficient context, say so clearly."
        )

    context_blocks: list[str] = []
    for i, rec in enumerate(context_records, start=1):
        block = f"""
--- Context Block {i} (score={rec.get('score', 0):.4f}) ---
Title: {rec.get('title', 'N/A')}
Type:  {rec.get('okf_type', 'N/A')}
Tags:  {', '.join(rec.get('tags') or [])}

OKF YAML Front-Matter:
```yaml
{rec.get('raw_yaml', '').strip()}
```

Body:
{rec.get('body', '').strip()}

Linked Concepts: {', '.join(rec.get('concepts') or []) or 'None'}
Sequential Neighbours: {', '.join(rec.get('next_blocks') or []) or 'None'}
"""
        context_blocks.append(block)

    context_str = "\n".join(context_blocks)
    return f"""You are an expert knowledge assistant powered by a GraphRAG system built on the Open Knowledge Format (OKF).

## Retrieved Context
The following OKF knowledge blocks were retrieved from the graph database using hybrid vector + graph traversal search. Use them as your primary source of truth.

{context_str}

## Instructions
- Ground your answers in the retrieved context above.
- When referencing a specific block, mention its title.
- If the context does not cover the question, clearly state that the information is not available in the current knowledge graph.
- Reason carefully before giving your final answer.
"""


# ── LiveThoughtCallbackHandler ─────────────────────────────────────────────────

class LiveThoughtCallbackHandler:
    """
    Intercepts DeepSeek's dual-stream output and yields typed SSE frames.

    Usage (inside an async generator):
        handler = LiveThoughtCallbackHandler(conversation_id)
        async for frame in handler.stream(client, messages):
            yield frame
    """

    def __init__(self, conversation_id: UUID) -> None:
        self.conversation_id = conversation_id
        self._thought_buffer: list[str] = []
        self._answer_buffer: list[str] = []

    def _sse(self, event_type: str, content: str) -> str:
        """Format a single SSE data line."""
        payload = json.dumps(
            {
                "event_type": event_type,
                "content": content,
                "conversation_id": str(self.conversation_id),
            }
        )
        return f"data: {payload}\n\n"

    async def stream(
        self,
        client: AsyncOpenAI,
        messages: list[dict],
    ) -> AsyncGenerator[str, None]:
        """
        Open a streaming completion request and yield SSE frames.

        Separates reasoning_content (thought) tokens from content (answer) tokens.
        """
        log.info(
            "Opening DeepSeek stream. conversation_id=%s model=%s",
            self.conversation_id,
            settings.DEEPSEEK_MODEL,
        )

        try:
            async with client.chat.completions.stream(
                model=settings.DEEPSEEK_MODEL,
                messages=messages,
                # DeepSeek-specific reasoning parameters (ignored by other providers)
                # max_tokens=8192,
            ) as stream:
                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # ── Reasoning / thinking tokens ────────────────────────────
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        self._thought_buffer.append(reasoning)
                        yield self._sse("thought", reasoning)

                    # ── Final answer tokens ────────────────────────────────────
                    answer = getattr(delta, "content", None) or ""
                    if answer:
                        self._answer_buffer.append(answer)
                        yield self._sse("answer", answer)

            # ── Stream finished ────────────────────────────────────────────────
            yield self._sse("done", "")
            log.info(
                "Stream complete. thought_tokens=%d answer_tokens=%d",
                len("".join(self._thought_buffer)),
                len("".join(self._answer_buffer)),
            )

        except Exception as exc:
            log.exception("Stream error: %s", exc)
            yield self._sse("error", f"Stream error: {exc}")
            # Do NOT re-raise here — the response has already started.
            # The error frame above notifies the client cleanly.


# ── Public API ─────────────────────────────────────────────────────────────────

async def stream_chat_response(
    user_message: str,
    context_records: list[dict],
    conversation_id: UUID | None = None,
) -> AsyncGenerator[str, None]:
    """
    Top-level async generator for the /chat route.

    1. Builds the system prompt with injected OKF context.
    2. Emits a 'context' frame summarising what was retrieved.
    3. Delegates token streaming to LiveThoughtCallbackHandler.

    Parameters
    ----------
    user_message      : str
    context_records   : list[dict]  — from graph_store.query_graph_context()
    conversation_id   : UUID | None — auto-assigned if None

    Yields
    ------
    str — SSE-formatted data frames
    """
    conv_id: UUID = conversation_id or uuid4()
    client = AsyncOpenAI(
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )

    system_prompt = _build_system_prompt(context_records)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    # Emit context summary frame before the LLM stream begins
    context_summary = f"Retrieved {len(context_records)} OKF context block(s) from the knowledge graph."
    context_frame = json.dumps(
        {
            "event_type": "context",
            "content": context_summary,
            "conversation_id": str(conv_id),
        }
    )
    yield f"data: {context_frame}\n\n"

    # Delegate to the callback handler
    handler = LiveThoughtCallbackHandler(conversation_id=conv_id)
    async for frame in handler.stream(client, messages):
        yield frame

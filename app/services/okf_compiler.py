"""
services/okf_compiler.py — Multi-agent OKF compilation pipeline (LangGraph + DeepSeek).

Architecture
------------
Three specialised async LLM agents run sequentially inside a LangGraph StateGraph
to transform raw Markdown into validated OKF data structures:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Raw Markdown                                                         │
  │       │                                                               │
  │       ▼                                                               │
  │  [Node 1 — extractor_node]   → identifies Entities & Concepts        │
  │       │                                                               │
  │       ▼                                                               │
  │  [Node 2 — synthesizer_node] → generates OKF Markdown summary        │
  │       │                                                               │
  │       ▼                                                               │
  │  [Node 3 — formatter_node]   → validates + prepends YAML front-matter│
  │       │                                                               │
  │       ▼                                                               │
  │  final_okf_yaml  (ready for Neo4j persistence)                        │
  └──────────────────────────────────────────────────────────────────────┘

Streaming:
  Each node calls DeepSeek with stream=True. If a LiveThoughtCallbackHandler
  is present in the `callbacks` list, reasoning_content (<thought>) tokens are
  forwarded to it in real-time, allowing the frontend to display "thinking" before
  the final answer arrives.
"""

from __future__ import annotations

import json
import re
import yaml
from dataclasses import dataclass, field
from typing import Any, TypedDict

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from app.config import get_settings
from app.exceptions import OKFCompilationError
from app.logger import get_logger

try:
    from langgraph.graph import StateGraph, END
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

log = get_logger(__name__)
settings = get_settings()

# ── Global DeepSeek Client (OpenAI-compatible) ─────────────────────────────────
client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
)


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    """A raw entity detected by the Extractor node."""
    name: str
    entity_type: str          # e.g. "concept", "person", "technology"
    source_excerpt: str


@dataclass
class OKFBlock:
    """
    A fully compiled and validated OKF knowledge block.

    Each block maps to a single Neo4j (OKFNode) and zero or more
    (Concept) nodes linked by [:REFERENCES] edges.
    """
    title: str
    okf_type: str
    tags: list[str]
    links: list[dict[str, str]]
    body: str
    yaml_valid: bool
    raw_yaml: str
    concepts: list[str] = field(default_factory=list)


# ── Pydantic Models for Structured LLM Output ─────────────────────────────────

class EntityItem(BaseModel):
    """A single extracted entity as returned by DeepSeek."""
    name: str = Field(..., description="Entity or concept name")
    entity_type: str = Field(..., description="One of: concept, person, technology, process, organization")
    source_excerpt: str = Field(..., description="Verbatim excerpt from source text that identifies this entity")


class ExtractorOutput(BaseModel):
    """Structured response expected from the Extractor node."""
    entities: list[EntityItem] = Field(..., description="All extracted entities and concepts")


class OKFFrontmatter(BaseModel):
    """Strict YAML front-matter schema for a compiled OKF block."""
    title: str = Field(..., description="Human-readable title of the OKF knowledge block")
    type: str = Field(..., description="Semantic type: concept | process | person | technology | organization")
    summary: str = Field(..., description="One-sentence summary of this block's content")
    tags: list[str] = Field(default_factory=list)
    links: list[dict[str, str]] = Field(default_factory=list)


# ── LangGraph State ────────────────────────────────────────────────────────────

class OKFState(TypedDict):
    """Shared mutable state threaded through the LangGraph pipeline."""
    raw_markdown: str
    extracted_entities: list[str]          # plain string list for inter-node transfer
    draft_okf_content: str                  # synthesiser output (Markdown)
    final_okf_yaml: str                     # formatter output (YAML fm + body)
    callbacks: list[Any]                    # LiveThoughtCallbackHandler instances


# ── Streaming Helper ───────────────────────────────────────────────────────────

async def _stream_completion(
    messages: list[dict],
    stage_label: str,
    callbacks: list[Any],
) -> str:
    """
    Call DeepSeek with stream=True, forward <thought> tokens to any registered
    LiveThoughtCallbackHandler, and return the fully accumulated answer string.

    Parameters
    ----------
    messages    : OpenAI-format message list
    stage_label : Human-readable label for log lines (e.g. "[Extractor]")
    callbacks   : List of handler objects that expose an `on_thought_token(str)` coroutine
    """
    thought_buffer: list[str] = []
    answer_buffer: list[str] = []

    try:
        stream = await client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # ── Forward reasoning/thought tokens to registered callbacks ──────
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                thought_buffer.append(reasoning)
                for cb in callbacks:
                    on_thought = getattr(cb, "on_thought_token", None)
                    if callable(on_thought):
                        try:
                            import inspect
                            if inspect.iscoroutinefunction(on_thought):
                                await on_thought(stage_label, reasoning)
                            else:
                                on_thought(stage_label, reasoning)
                        except Exception as cb_exc:  # never crash the pipeline on callback errors
                            log.warning("%s Callback error: %s", stage_label, cb_exc)

            # ── Accumulate final answer tokens ────────────────────────────────
            answer = getattr(delta, "content", None) or ""
            if answer:
                answer_buffer.append(answer)

    except Exception as exc:
        log.error("%s DeepSeek streaming error: %s", stage_label, exc)
        raise OKFCompilationError(f"{stage_label} LLM call failed: {exc}") from exc

    full_answer = "".join(answer_buffer)
    log.info(
        "%s Stream complete. thought_tokens=%d answer_chars=%d",
        stage_label,
        sum(len(t) for t in thought_buffer),
        len(full_answer),
    )
    return full_answer


# ── Node 1 — Extractor ────────────────────────────────────────────────────────

_EXTRACTOR_SYSTEM = """\
You are a Principal Knowledge Engineer specialising in entity and concept extraction.

Your task is to analyse the provided raw Markdown document and extract ALL meaningful:
  • Core concepts and ideas
  • Named entities (people, organisations, technologies, products)
  • Processes and methodologies
  • Key relationships between entities

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema — no prose, no markdown fences:
{
  "entities": [
    {
      "name": "<entity name>",
      "entity_type": "<one of: concept | person | technology | process | organization>",
      "source_excerpt": "<brief verbatim excerpt from the document identifying this entity>"
    }
  ]
}

Rules:
- Be exhaustive; prefer recall over precision.
- Deduplicate: each unique name appears at most once.
- Normalise names to title case.
- source_excerpt must be a direct quote from the input, max 120 characters.
"""

async def extractor_node(state: OKFState) -> OKFState:
    """
    LangGraph Node 1 — Entity Extraction.

    Sends raw_markdown to DeepSeek and parses the JSON response into a
    deduplicated list of entity name strings. Updates `extracted_entities`.
    """
    log.info("[Extractor] Starting entity extraction …")

    messages = [
        {"role": "system", "content": _EXTRACTOR_SYSTEM},
        {"role": "user",   "content": f"## Document\n\n{state['raw_markdown']}"},
    ]

    try:
        raw_response = await _stream_completion(
            messages=messages,
            stage_label="[Extractor]",
            callbacks=state.get("callbacks", []),
        )

        # Strip optional markdown fences the model may add despite instructions
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        parsed: ExtractorOutput = ExtractorOutput.model_validate_json(cleaned)
        entity_names = [e.name for e in parsed.entities]

        # Deduplicate preserving order
        seen: set[str] = set()
        unique_names: list[str] = []
        for name in entity_names:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                unique_names.append(name)

        log.info("[Extractor] Extracted %d unique entities.", len(unique_names))
        return {**state, "extracted_entities": unique_names}

    except ValidationError as ve:
        log.error("[Extractor] Pydantic validation failed: %s", ve)
        # Graceful degradation: fall back to empty list so pipeline can continue
        return {**state, "extracted_entities": []}
    except OKFCompilationError:
        raise
    except Exception as exc:
        log.exception("[Extractor] Unexpected error: %s", exc)
        raise OKFCompilationError(f"Extractor node failed: {exc}") from exc


# ── Node 2 — Synthesizer ──────────────────────────────────────────────────────

_SYNTHESIZER_SYSTEM = """\
You are a Senior Technical Writer specialising in the Open Knowledge Format (OKF).

Given a raw Markdown document and a list of extracted entities/concepts, write a
comprehensive, well-structured summary in Markdown format.

The summary MUST:
  1. Begin with a brief (2–3 sentence) executive overview.
  2. Contain a dedicated ## section for each major concept/entity in the list.
  3. Explain each entity's role, significance, and relationships to other entities.
  4. Preserve and cite important details, figures, and terminology from the source.
  5. End with a ## Key Relationships section describing how the entities interconnect.

Style guidelines:
  - Write in clear, precise, third-person technical prose.
  - Do NOT fabricate facts. Stay strictly grounded in the source document.
  - Use bullet points within sections for lists of attributes or relationships.
  - Total length: 300–800 words (scale with document complexity).

OUTPUT: Return ONLY the Markdown body — no YAML front-matter, no JSON wrappers.
"""

async def synthesizer_node(state: OKFState) -> OKFState:
    """
    LangGraph Node 2 — OKF Content Synthesis.

    Takes raw_markdown + extracted_entities and asks DeepSeek to produce a
    comprehensive, structured Markdown summary. Updates `draft_okf_content`.
    """
    log.info("[Synthesizer] Synthesizing OKF content for %d entities …", len(state["extracted_entities"]))

    entities_block = "\n".join(f"  - {e}" for e in state["extracted_entities"])
    user_content = (
        f"## Extracted Entities\n{entities_block}\n\n"
        f"## Source Document\n\n{state['raw_markdown']}"
    )

    messages = [
        {"role": "system", "content": _SYNTHESIZER_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    try:
        draft = await _stream_completion(
            messages=messages,
            stage_label="[Synthesizer]",
            callbacks=state.get("callbacks", []),
        )
        log.info("[Synthesizer] Draft content generated (%d chars).", len(draft))
        return {**state, "draft_okf_content": draft.strip()}

    except OKFCompilationError:
        raise
    except Exception as exc:
        log.exception("[Synthesizer] Unexpected error: %s", exc)
        raise OKFCompilationError(f"Synthesizer node failed: {exc}") from exc


# ── Node 3 — Formatter ────────────────────────────────────────────────────────

_FORMATTER_SYSTEM = """\
You are a Knowledge Architect responsible for producing valid OKF (Open Knowledge Format) documents.

Given a Markdown content block, prepend a strict YAML front-matter section that captures the
document's metadata. The front-matter MUST include these exact fields:

---
title: "<concise, descriptive title — sentence case>"
type: "<one of: concept | process | person | technology | organization>"
summary: "<single sentence summarising the entire document>"
tags:
  - "<tag1>"
  - "<tag2>"
links: []
---

Rules:
  1. The YAML block must be delimited by `---` on its own line at the top and bottom.
  2. `title`, `type`, and `summary` are REQUIRED and must be non-empty strings.
  3. `tags` should contain 3–6 relevant lowercase keywords derived from the content.
  4. `links` should be an empty list `[]` unless cross-references are explicit in the content.
  5. After the closing `---`, output the ORIGINAL Markdown content unchanged.
  6. Return ONLY the final document (YAML front-matter + Markdown body). No extra prose.
"""

async def formatter_node(state: OKFState) -> OKFState:
    """
    LangGraph Node 3 — YAML Front-matter Formatting & Validation.

    Takes draft_okf_content and asks DeepSeek to prepend a strict YAML front-matter
    block. Validates the generated YAML with Pydantic. Updates `final_okf_yaml`.
    """
    log.info("[Formatter] Formatting OKF document with YAML front-matter …")

    messages = [
        {"role": "system", "content": _FORMATTER_SYSTEM},
        {"role": "user",   "content": f"## Content\n\n{state['draft_okf_content']}"},
    ]

    try:
        formatted = await _stream_completion(
            messages=messages,
            stage_label="[Formatter]",
            callbacks=state.get("callbacks", []),
        )
        formatted = formatted.strip()

        # ── Validate the generated YAML front-matter via Pydantic ──────────────
        fm_match = re.match(r"^---\s*\n(.*?)\n---", formatted, re.DOTALL)
        if fm_match:
            try:
                fm_dict = yaml.safe_load(fm_match.group(1))
                OKFFrontmatter.model_validate(fm_dict)
                log.info("[Formatter] YAML front-matter validation passed.")
            except (yaml.YAMLError, ValidationError) as ve:
                log.warning("[Formatter] YAML validation warning: %s", ve)
                # Continue anyway — partial output is still usable downstream
        else:
            log.warning("[Formatter] No YAML front-matter block detected in formatter output.")

        log.info("[Formatter] Final OKF document ready (%d chars).", len(formatted))
        return {**state, "final_okf_yaml": formatted}

    except OKFCompilationError:
        raise
    except Exception as exc:
        log.exception("[Formatter] Unexpected error: %s", exc)
        raise OKFCompilationError(f"Formatter node failed: {exc}") from exc


# ── LangGraph Pipeline Builder ─────────────────────────────────────────────────

def _build_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Falls back to a simple sequential async runner if langgraph is not installed,
    keeping the public API identical.
    """
    if not _LANGGRAPH_AVAILABLE:
        log.warning("langgraph not installed — running pipeline in sequential fallback mode.")
        return None

    graph = StateGraph(OKFState)

    graph.add_node("extractor_node",   extractor_node)
    graph.add_node("synthesizer_node", synthesizer_node)
    graph.add_node("formatter_node",   formatter_node)

    graph.set_entry_point("extractor_node")
    graph.add_edge("extractor_node",   "synthesizer_node")
    graph.add_edge("synthesizer_node", "formatter_node")
    graph.add_edge("formatter_node",   END)

    return graph.compile()


# Eagerly compile at module load time (cheap, no I/O)
_compiled_graph = _build_graph()


# ── Public API ─────────────────────────────────────────────────────────────────

async def compile_markdown_to_okf(
    markdown_text: str,
    callbacks: list[Any] | None = None,
) -> str:
    """
    Orchestrate the three-node OKF compilation pipeline.

    Parameters
    ----------
    markdown_text : str
        Raw Markdown produced by parse_pdf_to_markdown().
    callbacks : list, optional
        List of handler objects (e.g. LiveThoughtCallbackHandler) that expose
        `on_thought_token(stage: str, token: str)` — called for every
        reasoning_content chunk emitted by DeepSeek.

    Returns
    -------
    str
        The final OKF document string: YAML front-matter + Markdown body,
        ready to be parsed and persisted to Neo4j.

    Raises
    ------
    OKFCompilationError
        If any node stage fails unrecoverably.
    """
    if not markdown_text or not markdown_text.strip():
        raise OKFCompilationError("compile_markdown_to_okf received empty markdown_text.")

    log.info("OKF compilation pipeline starting … (langgraph=%s)", _LANGGRAPH_AVAILABLE)

    initial_state: OKFState = {
        "raw_markdown":       markdown_text,
        "extracted_entities": [],
        "draft_okf_content":  "",
        "final_okf_yaml":     "",
        "callbacks":          callbacks or [],
    }

    try:
        if _compiled_graph is not None:
            # ── LangGraph execution path ───────────────────────────────────────
            final_state: OKFState = await _compiled_graph.ainvoke(initial_state)
        else:
            # ── Sequential fallback (no langgraph dependency) ──────────────────
            state = initial_state
            state = await extractor_node(state)
            state = await synthesizer_node(state)
            state = await formatter_node(state)
            final_state = state

    except OKFCompilationError:
        raise
    except Exception as exc:
        log.exception("Unexpected error in OKF compilation pipeline: %s", exc)
        raise OKFCompilationError(str(exc)) from exc

    result = final_state.get("final_okf_yaml", "")
    if not result:
        raise OKFCompilationError("Pipeline completed but final_okf_yaml is empty.")

    log.info("OKF compilation complete (%d chars).", len(result))
    return result


# ── Legacy Compatibility: OKF Block list builder ───────────────────────────────
# Retained so graph_store.persist_okf_blocks can still receive a list[OKFBlock]
# when called from the legacy upload route.

def parse_okf_yaml_to_blocks(final_okf_yaml: str) -> list[OKFBlock]:
    """
    Parse the compiler's final_okf_yaml string into a list[OKFBlock].

    The compiler now returns a single unified OKF document; this helper
    wraps it in a single-element list for compatibility with persist_okf_blocks.
    """
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", final_okf_yaml, re.DOTALL)
    if fm_match:
        try:
            fm = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError:
            fm = {}
        body = fm_match.group(2).strip()
    else:
        fm = {}
        body = final_okf_yaml.strip()

    raw_yaml_str = yaml.dump(
        {k: fm.get(k, "") for k in ("title", "type", "summary", "tags", "links")},
        default_flow_style=False,
    )

    return [
        OKFBlock(
            title=fm.get("title", "Untitled"),
            okf_type=fm.get("type", "concept"),
            tags=fm.get("tags", []),
            links=fm.get("links", []),
            body=body,
            yaml_valid=bool(fm.get("title") and fm.get("type") and fm.get("summary")),
            raw_yaml=raw_yaml_str,
            concepts=[
                lnk["target"]
                for lnk in (fm.get("links") or [])
                if isinstance(lnk, dict) and "target" in lnk
            ],
        )
    ]

"""
services/okf_compiler.py — Multi-agent OKF compilation pipeline.

Architecture
------------
Three specialised async agents run sequentially (with optional parallelism
in Stage 2) to transform raw Markdown into validated OKF data structures:

  ┌──────────────────────────────────────────────────────────────────────┐
  │  Raw Markdown                                                         │
  │       │                                                               │
  │       ▼                                                               │
  │  [Agent 1 — Extractor]   → identifies Entities & Concepts            │
  │       │                                                               │
  │       ▼                                                               │
  │  [Agent 2 — Synthesizer] → generates OKF Markdown blocks (YAML fm)  │
  │       │                                                               │
  │       ▼                                                               │
  │  [Agent 3 — Formatter]   → validates YAML front-matter               │
  │       │                                                               │
  │       ▼                                                               │
  │  List[OKFBlock]  (ready for Neo4j persistence)                        │
  └──────────────────────────────────────────────────────────────────────┘

Each agent is an async function that accepts the previous stage's output
and returns a typed dataclass, keeping the pipeline fully composable.
"""

from __future__ import annotations

import asyncio
import re
import yaml
from dataclasses import dataclass, field
from typing import Any

from app.exceptions import OKFCompilationError
from app.logger import get_logger

log = get_logger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class ExtractedEntity:
    """A raw entity detected by the Extractor agent."""
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


# ── Agent 1 — Extractor ────────────────────────────────────────────────────────

async def _run_extractor(markdown_text: str) -> list[ExtractedEntity]:
    """
    Agent 1: Entity Extraction

    Production: Call an LLM with a structured extraction prompt, requesting
    JSON output listing all entities, their types, and supporting text spans.

    Mock: Detect Markdown headings and bold terms as heuristic entities.
    """
    log.info("[Extractor] Starting entity extraction …")
    await asyncio.sleep(0.05)  # simulate LLM latency

    entities: list[ExtractedEntity] = []

    # Heuristic extraction from headings
    heading_pattern = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    for match in heading_pattern.finditer(markdown_text):
        title = match.group(1).strip()
        entities.append(
            ExtractedEntity(
                name=title,
                entity_type="concept",
                source_excerpt=title,
            )
        )

    # Heuristic extraction from bold terms
    bold_pattern = re.compile(r"\*\*(.+?)\*\*")
    for match in bold_pattern.finditer(markdown_text):
        term = match.group(1).strip()
        if len(term) > 2:
            entities.append(
                ExtractedEntity(
                    name=term,
                    entity_type="technology" if any(
                        kw in term.lower()
                        for kw in ["gnn", "rag", "llm", "embedding", "graph", "vector"]
                    ) else "concept",
                    source_excerpt=term,
                )
            )

    # Deduplicate by name
    seen: set[str] = set()
    unique_entities = []
    for e in entities:
        key = e.name.lower()
        if key not in seen:
            seen.add(key)
            unique_entities.append(e)

    log.info("[Extractor] Found %d unique entities.", len(unique_entities))
    return unique_entities


# ── Agent 2 — Synthesizer ─────────────────────────────────────────────────────

def _extract_section_body(entity: ExtractedEntity, markdown_text: str) -> str:
    """
    Extract the actual content from markdown that is relevant to a given entity.

    Strategy:
    - For heading-based entities: extract all text under that heading until the
      next heading of the same or higher level.
    - For bold-term entities: extract a context window (±3 paragraphs) around
      the first occurrence of the term.
    """
    name = entity.name
    lines = markdown_text.splitlines()

    # ── Strategy 1: Heading section extraction ─────────────────────────────────
    # Find lines that match "# Title" / "## Title" / "### Title"
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$")
    for i, line in enumerate(lines):
        m = heading_pattern.match(line)
        if m and m.group(2).strip().lower() == name.lower():
            level = len(m.group(1))  # number of # characters
            # Collect lines until the next heading of equal or higher level
            body_lines: list[str] = []
            for subsequent_line in lines[i + 1 :]:
                next_heading = heading_pattern.match(subsequent_line)
                if next_heading and len(next_heading.group(1)) <= level:
                    break
                body_lines.append(subsequent_line)

            body_text = "\n".join(body_lines).strip()
            if body_text:
                return f"## {name}\n\n{body_text}"

    # ── Strategy 2: Context window around bold/inline mention ──────────────────
    # Split into paragraphs and find the one(s) that mention the term
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", markdown_text) if p.strip()]
    mention_pattern = re.compile(re.escape(name), re.IGNORECASE)
    matching_indices = [i for i, p in enumerate(paragraphs) if mention_pattern.search(p)]

    if matching_indices:
        first = matching_indices[0]
        # Take up to 3 paragraphs around the first match
        start = max(0, first - 1)
        end = min(len(paragraphs), first + 3)
        excerpt = "\n\n".join(paragraphs[start:end])
        return f"## {name}\n\n{excerpt}"

    # ── Fallback: use the entity's source_excerpt ──────────────────────────────
    return f"## {name}\n\n{entity.source_excerpt}"


async def _run_synthesizer(
    entities: list[ExtractedEntity],
    markdown_text: str,
) -> list[dict[str, Any]]:
    """
    Agent 2: OKF Markdown Block Synthesis

    Extracts the real surrounding content from the source markdown for each
    entity, producing OKF blocks with substantive body text grounded in the
    original document. This ensures that Neo4j nodes contain actual knowledge
    rather than placeholder stubs.

    Production upgrade: Replace body extraction with an LLM call that
    summarises/rewrites the extracted excerpt into clean OKF prose.
    """
    log.info("[Synthesizer] Synthesizing %d OKF blocks …", len(entities))
    await asyncio.sleep(0.05)

    okf_raw_blocks: list[dict[str, Any]] = []

    for idx, entity in enumerate(entities):
        # Build cross-links to neighbouring entities
        links = []
        if idx + 1 < len(entities):
            links.append(
                {"target": entities[idx + 1].name, "relation": "NEXT"}
            )
        if idx > 0:
            links.append(
                {"target": entities[idx - 1].name, "relation": "PREV"}
            )

        # Extract the actual content from the markdown for this entity
        body = _extract_section_body(entity, markdown_text)

        okf_raw_blocks.append(
            {
                "title": entity.name,
                "okf_type": entity.entity_type,
                "tags": [entity.entity_type, "auto-generated"],
                "links": links,
                "body": body,
            }
        )

    log.info("[Synthesizer] Generated %d raw OKF blocks.", len(okf_raw_blocks))
    return okf_raw_blocks


# ── Agent 3 — Formatter & Validator ───────────────────────────────────────────

async def _run_formatter(raw_blocks: list[dict[str, Any]]) -> list[OKFBlock]:
    """
    Agent 3: YAML Validation & Formatting

    Production: Parse each block's YAML front-matter with a strict schema
    (e.g., using pydantic or jsonschema), fix common issues (missing tags,
    malformed links), and emit a structured OKFBlock ready for persistence.

    Mock: Serialize to YAML, parse back, and validate required keys.
    """
    log.info("[Formatter] Validating %d blocks …", len(raw_blocks))
    await asyncio.sleep(0.02)

    required_keys = {"title", "okf_type", "tags", "links"}
    compiled_blocks: list[OKFBlock] = []

    for raw in raw_blocks:
        front_matter_dict = {
            "title": raw.get("title", "Untitled"),
            "okf_type": raw.get("okf_type", "concept"),
            "tags": raw.get("tags", []),
            "links": raw.get("links", []),
        }

        # Serialize and immediately re-parse (round-trip validates YAML)
        raw_yaml = yaml.dump(front_matter_dict, default_flow_style=False)
        yaml_valid = False
        try:
            parsed = yaml.safe_load(raw_yaml)
            yaml_valid = all(k in parsed for k in required_keys)
        except yaml.YAMLError as exc:
            log.warning("[Formatter] YAML validation failed for '%s': %s", raw.get("title"), exc)

        block = OKFBlock(
            title=front_matter_dict["title"],
            okf_type=front_matter_dict["okf_type"],
            tags=front_matter_dict["tags"],
            links=front_matter_dict["links"],
            body=raw.get("body", ""),
            yaml_valid=yaml_valid,
            raw_yaml=raw_yaml,
            concepts=[
                lnk["target"]
                for lnk in front_matter_dict["links"]
                if isinstance(lnk, dict) and "target" in lnk
            ],
        )
        compiled_blocks.append(block)

    valid_count = sum(1 for b in compiled_blocks if b.yaml_valid)
    log.info(
        "[Formatter] Validation complete. %d/%d blocks valid.",
        valid_count,
        len(compiled_blocks),
    )
    return compiled_blocks


# ── Pipeline Orchestrator ──────────────────────────────────────────────────────

async def compile_markdown_to_okf(markdown_text: str) -> list[OKFBlock]:
    """
    Orchestrate the three-agent compilation pipeline.

    Parameters
    ----------
    markdown_text : str
        Raw Markdown produced by parse_pdf_to_markdown().

    Returns
    -------
    list[OKFBlock]
        Validated OKF blocks ready for Neo4j persistence.

    Raises
    ------
    OKFCompilationError
        If any agent stage fails.
    """
    log.info("OKF compilation pipeline starting …")

    try:
        # Stage 1 — Entity Extraction
        entities = await _run_extractor(markdown_text)
        if not entities:
            raise OKFCompilationError("Extractor returned zero entities.")

        # Stage 2 — OKF Block Synthesis
        raw_blocks = await _run_synthesizer(entities, markdown_text)

        # Stage 3 — YAML Validation & Formatting
        okf_blocks = await _run_formatter(raw_blocks)

    except OKFCompilationError:
        raise
    except Exception as exc:
        log.exception("Unexpected error in OKF compilation pipeline: %s", exc)
        raise OKFCompilationError(str(exc)) from exc

    log.info("OKF compilation complete. Total blocks: %d", len(okf_blocks))
    return okf_blocks

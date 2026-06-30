"""
services/pdf_parser.py — Abstract PDF-to-Markdown pipeline.

In production this would call LlamaParse (cloud) or a local PyMuPDF/pdfminer
pipeline. The async stub below demonstrates the exact interface contract that
the upload route depends on, making it trivially swappable.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.exceptions import PDFParseError
from app.logger import get_logger

log = get_logger(__name__)


async def parse_pdf_to_markdown(file_path: Path) -> tuple[str, int]:
    """
    Convert a PDF on disk to a raw Markdown string.

    Parameters
    ----------
    file_path : Path
        Absolute path to the uploaded PDF on the local filesystem.

    Returns
    -------
    tuple[str, int]
        (markdown_text, page_count)

    Raises
    ------
    PDFParseError
        If the file cannot be read or parsed.

    Production replacement
    ----------------------
    Replace the stub below with:

        from llama_parse import LlamaParse
        parser = LlamaParse(result_type="markdown", verbose=True)
        documents = await parser.aload_data(str(file_path))
        markdown_text = "\\n\\n".join(doc.text for doc in documents)
        page_count = len(documents)
        return markdown_text, page_count
    """
    log.info("Parsing PDF: %s", file_path)

    if not file_path.exists():
        raise PDFParseError(f"File not found at path: {file_path}")

    # ── Simulated async I/O latency (remove in production) ───────────────────
    await asyncio.sleep(0.1)

    # ── Mock output that mirrors real LlamaParse Markdown structure ───────────
    mock_markdown = f"""---
title: Parsed Document
source: {file_path.name}
---

# Chapter 1 — Introduction to Knowledge Graphs

Knowledge graphs encode information as a set of **entities** (nodes) and
**relationships** (edges). When combined with vector similarity search,
they enable **GraphRAG** — retrieval-augmented generation that is both
semantically rich and topologically aware.

## 1.1 Core Concepts

- **Node**: An entity (Person, Organisation, Concept, Document …)
- **Edge**: A directed relationship between two nodes
- **Property**: A key-value attribute on a node or edge
- **Embedding**: A dense vector representation of text

## 1.2 Open Knowledge Format (OKF)

OKF is a YAML-front-matter Markdown standard that encodes each block of
knowledge with:

```yaml
okf_type: concept
title: "Graph Neural Networks"
tags: [ml, graph, deep-learning]
links:
  - target: "Embeddings"
    relation: USES
  - target: "Node Classification"
    relation: ENABLES
```

Each block can carry free-form prose beneath the front matter.

# Chapter 2 — Vector Embeddings

Embeddings map text to a high-dimensional vector space …
"""

    # Rough page count from section headers
    page_count = mock_markdown.count("# Chapter")

    log.info("PDF parsed successfully. Estimated pages: %d", page_count)
    return mock_markdown, page_count

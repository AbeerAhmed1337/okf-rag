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

    try:
        import pymupdf4llm 
        import fitz
    except ImportError as exc:
        raise PDFParseError(
            "pymupdf4llm is not installed. Run: pip install pymupdf4llm"
        ) from exc

    if not file_path.exists():
        raise PDFParseError(f"File not found at path: {file_path}")

    try:
        # Run the CPU-bound PDF parsing in a thread pool to avoid blocking async
        loop = asyncio.get_event_loop()
        
        # 1. Convert PDF to Markdown
        markdown_text = await loop.run_in_executor(
            None, 
            lambda: pymupdf4llm.to_markdown(str(file_path))
        )
        
        # 2. Get accurate page count
        def _get_page_count():
            with fitz.open(str(file_path)) as doc:
                return len(doc)
                
        page_count = await loop.run_in_executor(None, _get_page_count)
        
        log.info("PDF parsed successfully. Pages: %d, Length: %d chars", page_count, len(markdown_text))
        return markdown_text, page_count
        
    except Exception as exc:
        log.exception("Failed to parse PDF: %s", exc)
        raise PDFParseError(f"PDF Parsing failed: {exc}") from exc

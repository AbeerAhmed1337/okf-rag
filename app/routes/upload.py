"""
routes/upload.py — POST /api/v1/upload

Ingest pipeline:
  1. Validate & stream the binary PDF into MongoDB GridFS (motor).
  2. Save a local temp copy for text extraction.
  3. Parse the PDF to Markdown (pymupdf4llm).
  4. Run the 3-node LangGraph OKF compilation pipeline (DeepSeek streaming).
  5. Parse the resulting YAML front-matter + body.
  6. Persist OKFNode + Concept nodes to Neo4j via ingest_okf_node().
  7. Return 202 Accepted with mongodb file_id and success metadata.
"""

from __future__ import annotations

import io
import re
import uuid
import yaml
from pathlib import Path
from typing import Any, AsyncGenerator

import aiofiles
from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import get_settings, Settings
from app.exceptions import FileTooLargeError, OKFCompilationError
from app.logger import get_logger
from app.schemas.upload import OKFNodeMeta, UploadResponse, UploadStatus
from app.services.embedding import generate_embeddings_batch
from app.services.graph_store import persist_okf_blocks
from app.services.neo4j_client import neo4j_client
from app.services.okf_compiler import (
    OKFBlock,
    compile_markdown_to_okf,
    parse_okf_yaml_to_blocks,
)
from app.services.pdf_parser import parse_pdf_to_markdown

log = get_logger(__name__)
router = APIRouter()


# ── Settings Dependency ────────────────────────────────────────────────────────

def _get_settings() -> Settings:
    return get_settings()


# ── MongoDB / GridFS Helper ────────────────────────────────────────────────────

async def _store_pdf_in_gridfs(
    file_data: bytes,
    filename: str,
    settings: Settings,
) -> str:
    """
    Stream binary PDF bytes into MongoDB GridFS using motor (async).

    Returns
    -------
    str
        The GridFS file_id as a hex string.

    Notes
    -----
    Motor and GridFS are optional — if MONGO_URI is not configured or motor is
    not installed, this step is skipped gracefully and an empty string is returned.
    The rest of the pipeline continues unaffected.
    """
    mongo_uri: str = getattr(settings, "MONGO_URI", "")
    if not mongo_uri:
        log.warning("MONGO_URI not configured — skipping GridFS storage.")
        return ""

    try:
        import motor.motor_asyncio as motor_asyncio  # type: ignore
        from motor.motor_asyncio import AsyncIOMotorGridFSBucket  # type: ignore

        motor_client = motor_asyncio.AsyncIOMotorClient(mongo_uri)
        db_name: str = getattr(settings, "MONGO_DB_NAME", "okf_rag")
        db = motor_client[db_name]
        bucket = AsyncIOMotorGridFSBucket(db, bucket_name="pdfs")

        file_id = await bucket.upload_from_stream(
            filename=filename,
            source=io.BytesIO(file_data),
            metadata={"content_type": "application/pdf"},
        )

        motor_client.close()
        hex_id = str(file_id)
        log.info("PDF stored in GridFS. file_id=%s filename=%s", hex_id, filename)
        return hex_id

    except ImportError:
        log.warning("motor is not installed — skipping GridFS storage. pip install motor")
        return ""
    except Exception as exc:
        # GridFS failure must NOT abort ingestion; log and continue.
        log.error("GridFS upload failed: %s", exc)
        return ""


# ── Local Temp-File Save ───────────────────────────────────────────────────────

async def _save_upload(file_data: bytes, filename: str, settings: Settings) -> Path:
    """Write the already-read bytes to a secure temp directory for PDF parsing."""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{uuid.uuid4().hex}_{Path(filename or 'upload.pdf').name}"
    dest_path = upload_dir / safe_name

    async with aiofiles.open(dest_path, "wb") as out_file:
        await out_file.write(file_data)

    log.info("Temp file saved: %s (%d bytes)", dest_path, len(file_data))
    return dest_path


# ── YAML Front-matter Parser ───────────────────────────────────────────────────

def _split_okf_document(okf_yaml_str: str) -> tuple[dict[str, Any], str]:
    """
    Split the compiler's output into (frontmatter_dict, markdown_body).

    The compiler outputs:
        ---
        title: "…"
        type:  "…"
        summary: "…"
        ---
        # Markdown body …

    Returns a (metadata dict, body string) tuple.
    """
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", okf_yaml_str, re.DOTALL)
    if fm_match:
        try:
            metadata = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as exc:
            log.warning("YAML front-matter parse error: %s", exc)
            metadata = {}
        body = fm_match.group(2).strip()
    else:
        log.warning("No YAML front-matter detected — treating entire output as body.")
        metadata = {}
        body = okf_yaml_str.strip()

    return metadata, body


# ── Neo4j Ingestion Wrapper ────────────────────────────────────────────────────

async def ingest_okf_node(
    metadata: dict[str, Any],
    content: str,
    okf_blocks: list[OKFBlock],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    """
    Persist compiled OKF blocks to Neo4j.

    This async wrapper isolates the session lifecycle so the route handler
    stays clean. It delegates to persist_okf_blocks() from graph_store.

    Parameters
    ----------
    metadata   : Parsed YAML front-matter dict (title, type, summary, …)
    content    : Markdown body string
    okf_blocks : list[OKFBlock] produced by parse_okf_yaml_to_blocks()
    embeddings : Embedding vectors aligned with okf_blocks

    Returns
    -------
    list[dict]  — persisted node metadata records from Neo4j
    """
    async with neo4j_client.session() as session:
        persisted = await persist_okf_blocks(session, okf_blocks, embeddings)

    log.info(
        "Neo4j ingestion complete. title=%s nodes=%d",
        metadata.get("title", "?"),
        len(persisted),
    )
    return persisted


# ── LiveThought Callback for Upload Pipeline ───────────────────────────────────

class _UploadThoughtLogger:
    """
    Minimal LiveThoughtCallbackHandler compatible shim for the upload route.

    During document ingestion there is no SSE response stream, so we simply
    log thought tokens at DEBUG level. Swap this for a WebSocket/SSE emitter
    if you want live compilation progress streamed to the client.
    """

    async def on_thought_token(self, stage: str, token: str) -> None:
        log.debug("LLM thought %s: %s", stage, token[:80])


# ── Upload Endpoint ────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
    summary="Upload a PDF and trigger OKF knowledge compilation",
    responses={
        202: {"description": "PDF accepted, OKF compilation and Neo4j ingestion complete"},
        413: {"description": "File too large"},
        422: {"description": "PDF parse failure or empty document"},
        500: {"description": "OKF pipeline or persistence error"},
    },
)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF document to ingest"),
    settings: Settings = Depends(_get_settings),
) -> JSONResponse:
    """
    **PDF Ingest Endpoint**

    Accepts a multipart PDF upload, runs it through the full OKF compilation
    pipeline (Entity Extraction → Synthesis → Formatting), and persists the
    resulting graph nodes to Neo4j. The raw binary PDF is also archived to
    MongoDB GridFS for audit/replay purposes.
    """
    task_id = uuid.uuid4()
    log.info("Upload received. task_id=%s filename=%s", task_id, file.filename)

    file_path: Path | None = None

    try:
        # ── 1. Read file bytes once (used for both GridFS and local save) ─────
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        chunks: list[bytes] = []
        bytes_read = 0

        while chunk := await file.read(65536):
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                raise FileTooLargeError(
                    f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB limit."
                )
            chunks.append(chunk)

        file_data = b"".join(chunks)
        log.info("Read %d bytes for task_id=%s", len(file_data), task_id)

        # ── 2. Stream binary PDF into MongoDB GridFS (non-blocking) ───────────
        gridfs_file_id: str = await _store_pdf_in_gridfs(
            file_data=file_data,
            filename=file.filename or "upload.pdf",
            settings=settings,
        )

        # ── 3. Save local temp copy for PDF text extraction ───────────────────
        file_path = await _save_upload(file_data, file.filename or "upload.pdf", settings)

        # ── 4. Parse PDF → Markdown ───────────────────────────────────────────
        markdown_text, page_count = await parse_pdf_to_markdown(file_path)
        log.info("PDF parsed. task_id=%s pages=%d chars=%d", task_id, page_count, len(markdown_text))

        # ── 5. LangGraph OKF Compilation Pipeline (DeepSeek streaming) ────────
        thought_logger = _UploadThoughtLogger()
        final_okf_yaml: str = await compile_markdown_to_okf(
            markdown_text=markdown_text,
            callbacks=[thought_logger],
        )
        log.info("OKF compiled. task_id=%s output_chars=%d", task_id, len(final_okf_yaml))

        # ── 6. Parse YAML front-matter + Markdown body ────────────────────────
        metadata, body_content = _split_okf_document(final_okf_yaml)
        log.info(
            "OKF document parsed. task_id=%s title=%s type=%s",
            task_id,
            metadata.get("title", "?"),
            metadata.get("type", "?"),
        )

        # ── 7. Convert to OKFBlock list (Neo4j schema) ────────────────────────
        okf_blocks: list[OKFBlock] = parse_okf_yaml_to_blocks(final_okf_yaml)

        # ── 8. Generate embeddings for OKF blocks ─────────────────────────────
        block_texts = [f"{b.title}\n{b.body}" for b in okf_blocks]
        embeddings = await generate_embeddings_batch(block_texts)
        log.info("Embeddings generated. task_id=%s count=%d", task_id, len(embeddings))

        # ── 9. Persist to Neo4j via ingest_okf_node ───────────────────────────
        persisted = await ingest_okf_node(
            metadata=metadata,
            content=body_content,
            okf_blocks=okf_blocks,
            embeddings=embeddings,
        )
        log.info("Persisted to Neo4j. task_id=%s nodes=%d", task_id, len(persisted))

        # ── 10. Build response ─────────────────────────────────────────────────
        okf_node_metas = [
            OKFNodeMeta(
                node_id=p["node_id"],
                label=p.get("label", "OKFNode"),
                title=p["title"],
                okf_type=p["okf_type"],
                yaml_valid=p["yaml_valid"],
            )
            for p in persisted
        ]

        concepts_total = sum(len(b.concepts) for b in okf_blocks)

        response = UploadResponse(
            task_id=task_id,
            status=UploadStatus.ACCEPTED,
            filename=file.filename or "unknown.pdf",
            page_count=page_count,
            okf_nodes=okf_node_metas,
            concepts_extracted=concepts_total,
            message=(
                f"PDF ingested successfully. "
                f"MongoDB file_id: {gridfs_file_id or 'not stored'}. "
                f"{len(persisted)} OKF node(s) persisted to Neo4j."
            ),
        )

        log.info(
            "Upload complete. task_id=%s nodes=%d concepts=%d file_id=%s",
            task_id, len(okf_node_metas), concepts_total, gridfs_file_id,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=response.model_dump(mode="json"),
        )

    finally:
        # Always clean up the temp file regardless of success or failure
        if file_path and file_path.exists():
            file_path.unlink(missing_ok=True)
            log.debug("Temp file removed: %s", file_path)

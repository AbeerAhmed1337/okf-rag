"""
routes/upload.py — POST /api/v1/upload

Ingest pipeline:
  1. Validate & save the uploaded PDF to a temporary directory.
  2. Parse the PDF to Markdown (async, via LlamaParse or mock).
  3. Run the 3-agent OKF compilation pipeline.
  4. Generate vector embeddings for each OKF block.
  5. Persist OKFNode + Concept nodes to Neo4j.
  6. Return 202 Accepted with task metadata.
"""

from __future__ import annotations

import os
import uuid
import aiofiles
from pathlib import Path

from fastapi import APIRouter, Depends, File, Request, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import get_settings, Settings
from app.exceptions import FileTooLargeError, OKFCompilationError
from app.logger import get_logger
from app.schemas.upload import OKFNodeMeta, UploadResponse, UploadStatus
from app.services.embedding import generate_embeddings_batch
from app.services.graph_store import persist_okf_blocks
from app.services.neo4j_client import neo4j_client
from app.services.okf_compiler import compile_markdown_to_okf, OKFBlock
from app.services.pdf_parser import parse_pdf_to_markdown

log = get_logger(__name__)
router = APIRouter()


def _get_settings() -> Settings:
    return get_settings()


async def _save_upload(file: UploadFile, settings: Settings) -> Path:
    """Stream-write the uploaded file to a secure temp directory."""
    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Unique filename to avoid collisions
    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename or 'upload.pdf').name}"
    dest_path = upload_dir / safe_name

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    bytes_written = 0

    async with aiofiles.open(dest_path, "wb") as out_file:
        while chunk := await file.read(65536):  # 64 KiB chunks
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                await out_file.close()
                dest_path.unlink(missing_ok=True)
                raise FileTooLargeError(
                    f"File exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit."
                )
            await out_file.write(chunk)

    log.info("Saved upload: %s (%d bytes)", dest_path, bytes_written)
    return dest_path


@router.post(
    "/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadResponse,
    summary="Upload a PDF and trigger OKF knowledge compilation",
    responses={
        202: {"description": "PDF accepted, processing initiated"},
        413: {"description": "File too large"},
        422: {"description": "PDF parse failure"},
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
    pipeline, and persists the resulting graph nodes to Neo4j.

    - **202 Accepted** — all steps completed; returns task metadata.
    - Background task support (Celery/ARQ) can be added by offloading
      stages 2-5 to a worker and returning immediately with a task_id.
    """
    task_id = uuid.uuid4()
    log.info("Upload received. task_id=%s filename=%s", task_id, file.filename)

    # ── 1. Validate content type ───────────────────────────────────────────────
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        log.warning("Rejected non-PDF upload: %s", file.content_type)
        # Allow octet-stream for clients that don't set MIME correctly

    # ── 2. Save to disk ───────────────────────────────────────────────────────
    file_path = await _save_upload(file, settings)

    try:
        # ── 3. Parse PDF → Markdown ───────────────────────────────────────────
        markdown_text, page_count = await parse_pdf_to_markdown(file_path)

        # ── 4. Multi-agent OKF compilation ────────────────────────────────────
        okf_blocks: list[OKFBlock] = await compile_markdown_to_okf(markdown_text)

        # ── 5. Generate embeddings for all blocks (batch call) ────────────────
        block_texts = [f"{b.title}\n{b.body}" for b in okf_blocks]
        embeddings = await generate_embeddings_batch(block_texts)

        # ── 6. Persist to Neo4j ───────────────────────────────────────────────
        async with neo4j_client.session() as session:
            persisted = await persist_okf_blocks(session, okf_blocks, embeddings)

        # ── 7. Build response metadata ────────────────────────────────────────
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
        )

        log.info(
            "Upload pipeline complete. task_id=%s nodes=%d concepts=%d",
            task_id,
            len(okf_node_metas),
            concepts_total,
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=response.model_dump(mode="json"),
        )

    finally:
        # Always clean up the temp file to avoid disk exhaustion
        if file_path.exists():
            file_path.unlink(missing_ok=True)
            log.debug("Temp file removed: %s", file_path)

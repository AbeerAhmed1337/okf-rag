"""
services/graph_store.py — Neo4j persistence layer for OKF nodes and concepts.

Graph Schema
------------
(OKFNode {id, title, okf_type, tags, body, raw_yaml, yaml_valid, embedding})
(Concept  {name})

Relationships:
  (OKFNode)-[:REFERENCES]->(Concept)   — entity links extracted from OKF
  (OKFNode)-[:NEXT]->(OKFNode)         — sequential block ordering

Indexes (expected to exist in Neo4j):
  CREATE VECTOR INDEX okfnode_embedding IF NOT EXISTS
    FOR (n:OKFNode) ON (n.embedding)
    OPTIONS { indexConfig: { `vector.dimensions`: 1536,
                             `vector.similarity_function`: 'cosine' } };
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from neo4j import AsyncSession

from app.exceptions import Neo4jPersistenceError
from app.logger import get_logger
from app.services.okf_compiler import OKFBlock

log = get_logger(__name__)


# ── Write Queries ──────────────────────────────────────────────────────────────

_MERGE_OKF_NODE = """
MERGE (n:OKFNode {title: $title})
SET   n.id        = coalesce(n.id, $id),
      n.okf_type  = $okf_type,
      n.tags      = $tags,
      n.body      = $body,
      n.raw_yaml  = $raw_yaml,
      n.yaml_valid = $yaml_valid,
      n.embedding  = $embedding
RETURN n.id AS node_id, labels(n)[0] AS label
"""

_MERGE_CONCEPT = """
MERGE (c:Concept {name: $name})
RETURN c.name AS name
"""

_MERGE_REFERENCES = """
MATCH (n:OKFNode {id: $node_id})
MATCH (c:Concept {name: $concept_name})
MERGE (n)-[:REFERENCES]->(c)
"""

_MERGE_NEXT_LINK = """
MATCH (a:OKFNode {id: $id_a})
MATCH (b:OKFNode {id: $id_b})
MERGE (a)-[:NEXT]->(b)
"""

# ── Hybrid Retrieval Queries ───────────────────────────────────────────────────

HYBRID_RETRIEVAL_CYPHER = """
// 1. Vector similarity search
CALL db.index.vector.queryNodes('okfnode_embedding', $top_k, $query_embedding)
YIELD node AS n, score

// 2. Pull referenced Concept nodes
OPTIONAL MATCH (n)-[:REFERENCES]->(c:Concept)

// 3. Pull sequential neighbours
OPTIONAL MATCH (n)-[:NEXT]->(next_block:OKFNode)

RETURN
    n.id          AS node_id,
    n.title       AS title,
    n.okf_type    AS okf_type,
    n.body        AS body,
    n.raw_yaml    AS raw_yaml,
    n.tags        AS tags,
    score,
    collect(DISTINCT c.name)        AS concepts,
    collect(DISTINCT next_block.title) AS next_blocks
ORDER BY score DESC
LIMIT $top_k
"""


# ── Persistence Helpers ────────────────────────────────────────────────────────

async def persist_okf_blocks(
    session: AsyncSession,
    blocks: list[OKFBlock],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    """
    Write all OKFBlock objects to Neo4j.

    Parameters
    ----------
    session    : Neo4j AsyncSession
    blocks     : Compiled OKF blocks from the compilation pipeline
    embeddings : Pre-computed embedding vectors (same order as `blocks`)

    Returns
    -------
    list[dict] — node_id + label for each persisted node
    """
    if len(blocks) != len(embeddings):
        raise Neo4jPersistenceError(
            f"Mismatch: {len(blocks)} blocks but {len(embeddings)} embeddings."
        )

    node_ids: list[str] = []
    persisted: list[dict[str, Any]] = []

    # ── Pass 1: Upsert OKFNode nodes ──────────────────────────────────────────
    for block, emb in zip(blocks, embeddings):
        node_id = str(uuid4())
        result = await session.run(
            _MERGE_OKF_NODE,
            id=node_id,
            title=block.title,
            okf_type=block.okf_type,
            tags=json.dumps(block.tags),
            body=block.body,
            raw_yaml=block.raw_yaml,
            yaml_valid=block.yaml_valid,
            embedding=emb,
        )
        record = await result.single()
        if record:
            persisted.append(
                {
                    "node_id": record["node_id"],
                    "label": record["label"],
                    "title": block.title,
                    "okf_type": block.okf_type,
                    "yaml_valid": block.yaml_valid,
                }
            )
            node_ids.append(record["node_id"])
        log.debug("Persisted OKFNode: %s", block.title)

    # ── Pass 2: Upsert Concept nodes & REFERENCES edges ───────────────────────
    for block, node_id in zip(blocks, node_ids):
        for concept_name in block.concepts:
            await session.run(_MERGE_CONCEPT, name=concept_name)
            await session.run(
                _MERGE_REFERENCES,
                node_id=node_id,
                concept_name=concept_name,
            )

    # ── Pass 3: Wire sequential NEXT edges ────────────────────────────────────
    for i in range(len(node_ids) - 1):
        await session.run(
            _MERGE_NEXT_LINK,
            id_a=node_ids[i],
            id_b=node_ids[i + 1],
        )

    log.info("Persisted %d OKFNode nodes to Neo4j.", len(persisted))
    return persisted


async def query_graph_context(
    session: AsyncSession,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Run the hybrid vector + graph traversal retrieval query.

    Returns a list of ranked context records each containing the OKFNode's
    body text, YAML, linked concepts, and sequential neighbours.
    """
    result = await session.run(
        HYBRID_RETRIEVAL_CYPHER,
        query_embedding=query_embedding,
        top_k=top_k,
    )
    records = await result.data()
    log.info("Graph context retrieval returned %d records.", len(records))
    return records

"""
Schema RAG Service — Neo4j AuraDB backend
──────────────────────────────────────────
Per-domain ``SchemaKnowledgeGraph`` registry backed by Neo4j AuraDB.

Lifecycle
---------
  • ``init_driver(driver, database)``  — called once from FastAPI lifespan
    after the Neo4j driver is opened and the vector index is ensured.

  • ``get_schema_context(...)``        — lazy per-domain KG build on first
    request; subsequent calls retrieve directly from Neo4j.

  • ``invalidate_cache(domain_id)``    — deletes the domain's nodes from
    Neo4j and marks it for rebuild on the next request.

Fallback
--------
  If the domain database (Postgres) is unreachable during the first build,
  the manifest's ``tables`` block is used as the schema source.

Public API
----------
    from app.services.schema_rag import init_driver, get_schema_context, invalidate_cache
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from neo4j import AsyncDriver

from app.services.schema_extractor import extract_schema
from app.services.schema_knowledge_graph import SchemaKnowledgeGraph

logger = logging.getLogger(__name__)


# ── Module-level singletons ───────────────────────────────────────────────────

_kg:           SchemaKnowledgeGraph | None = None
_domain_built: set[str]                   = set()
_build_locks:  dict[str, asyncio.Lock]    = {}


def init_driver(driver: AsyncDriver, database: str = "neo4j") -> None:
    """
    Initialise the module with an open Neo4j driver.

    Must be called once during application startup (FastAPI ``lifespan``)
    *after* ``ensure_vector_index()`` has been awaited.
    """
    global _kg
    _kg = SchemaKnowledgeGraph(driver, database)
    logger.info("Schema RAG: Neo4j driver initialised (database='%s')", database)


def _get_kg() -> SchemaKnowledgeGraph:
    if _kg is None:
        raise RuntimeError(
            "Neo4j driver not initialised. Call init_driver() in the FastAPI lifespan."
        )
    return _kg


# ── Public helpers ────────────────────────────────────────────────────────────

async def get_schema_context(
    domain_id: str,
    manifest: dict[str, Any],
    query: str,
    top_k: int = 10,
) -> str:
    """
    Return a ``[KNOWLEDGE GRAPH — RELEVANT SCHEMA]`` block for injection into
    the SQL-generation prompt.

    The KG for each domain is built lazily on first call (async
    double-checked locking) and stored in Neo4j for the lifetime of the
    process.  If the domain's live database is unreachable the manifest
    ``tables`` block is used as a schema source.

    Parameters
    ----------
    domain_id:
        Domain identifier (e.g. ``"sales"``).
    manifest:
        Fully-loaded and validated domain manifest dict.
    query:
        Natural-language question; used to rank relevant schema nodes.
    top_k:
        Number of closest nodes returned by the ANN vector search before
        graph expansion.

    Returns
    -------
    str
        Formatted schema context, or ``""`` if the KG could not be built or
        no relevant nodes were found.
    """
    kg = _get_kg()

    if domain_id not in _build_locks:
        _build_locks[domain_id] = asyncio.Lock()

    if domain_id not in _domain_built:
        async with _build_locks[domain_id]:
            # Double-checked locking: another coroutine may have built it
            # while we waited for the lock.
            if domain_id not in _domain_built:
                await _build_graph(kg, domain_id, manifest)
                _domain_built.add(domain_id)

    try:
        return await kg.retrieve(query, domain_id, top_k=top_k)
    except Exception as exc:
        logger.warning(
            "KG retrieval failed for domain '%s': %s", domain_id, exc
        )
        return ""


async def invalidate_cache(domain_id: str) -> None:
    """
    Delete *domain_id*'s nodes from Neo4j and mark the domain for a full
    rebuild on the next request.  Call after schema migrations or manifest
    updates.
    """
    kg = _get_kg()

    lock = _build_locks.get(domain_id)
    if lock:
        async with lock:
            await kg.delete_domain(domain_id)
            _domain_built.discard(domain_id)
    else:
        await kg.delete_domain(domain_id)
        _domain_built.discard(domain_id)

    logger.info("Schema KG cache invalidated for domain '%s'", domain_id)


# ── Internal ──────────────────────────────────────────────────────────────────

async def _build_graph(
    kg: SchemaKnowledgeGraph,
    domain_id: str,
    manifest: dict[str, Any],
) -> None:
    """
    Build the Neo4j KG for *domain_id*.

    Attempts to extract the live schema from Postgres first; falls back to
    the manifest ``tables`` block if the database is unreachable.
    """
    db_conn = manifest.get("database_connection", {})

    try:
        logger.info(
            "Domain '%s': extracting live schema from Postgres for KG build",
            domain_id,
        )
        schema = await extract_schema(db_conn)
        glossary    = manifest.get("business_glossary", {}) or {}
        sensitivity = manifest.get("sensitivity", []) or []
        await kg.build(schema, domain_id, glossary=glossary, sensitivity=sensitivity)
        logger.info(
            "Domain '%s': KG built from live DB (%d tables)",
            domain_id,
            len(schema.get("tables", [])),
        )
    except Exception as exc:
        logger.warning(
            "Domain '%s': live DB unavailable (%s). "
            "Building KG from manifest tables block.",
            domain_id,
            exc,
        )
        glossary    = manifest.get("business_glossary", {}) or {}
        sensitivity = manifest.get("sensitivity", []) or []
        fallback_schema = {"tables": manifest.get("tables", [])}
        await kg.build(fallback_schema, domain_id, glossary=glossary, sensitivity=sensitivity)
        logger.info(
            "Domain '%s': KG built from manifest fallback (%d tables)",
            domain_id,
            len(fallback_schema["tables"]),
        )

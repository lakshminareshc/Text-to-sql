"""
Routes: /api/v1/ingest

ONE endpoint.  One job:

  POST /api/v1/ingest/{domain_id}

  1. Load manifests/{domain_id}.yaml
     — contains business rules, glossary, sensitivity, few-shot examples

  2. Connect to the Postgres database named in that manifest
     — extract every table, column, PK, FK, and pg_description comment

  3. Merge schema into the manifest
     — manifest already has a `tables:` block; live DB columns overwrite it
       so the knowledge graph always reflects the real database

  4. Generate LLM descriptions for any table/column with a blank description
     — only calls the LLM for blanks; existing pg_description comments kept

  5. Write everything to Neo4j AuraDB as a knowledge graph
     — TABLE nodes + COLUMN nodes + HAS_COLUMN / REFERENCES / RELATED_TO edges
     — each node is embedded with text-embedding-3-large for semantic search

  DELETE /api/v1/ingest/{domain_id}
     — wipes the domain's nodes from Neo4j; next query auto-rebuilds
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import pathlib
import time

from fastapi import APIRouter, HTTPException

from app.manifests.manifest_loader import ManifestNotFoundError, get_manifest_loader
from app.models.schemas import IngestResponse, SchemaPreviewResponse
from app.services.schema_extractor import enrich_descriptions, extract_schema
from app.services.schema_rag import _build_locks, _domain_built, _get_kg, invalidate_cache

logger = logging.getLogger(__name__)
router = APIRouter()

_LOG_DIR = pathlib.Path(__file__).parent.parent.parent / "logs"


def _write_audit_log(
    domain_id: str,
    tables_ingested: int,
    columns_ingested: int,
    embeddings_generated: int,
    duration_ms: int,
) -> None:
    """Append one JSON-Lines record to logs/ingest_audit.jsonl."""
    try:
        _LOG_DIR.mkdir(exist_ok=True)
        record = {
            "timestamp":           datetime.datetime.utcnow().isoformat() + "Z",
            "domain_id":           domain_id,
            "tables_ingested":     tables_ingested,
            "columns_ingested":    columns_ingested,
            "embeddings_generated": embeddings_generated,
            "duration_ms":         duration_ms,
        }
        with (_LOG_DIR / "ingest_audit.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info(
            "[%s] Audit log written — %d ms, %d embeddings",
            domain_id, duration_ms, embeddings_generated,
        )
    except Exception as exc:  # never let logging crash the request
        logger.warning("[%s] Audit log write failed: %s", domain_id, exc)


@router.get(
    "/ingest/{domain_id}/schema",
    response_model=SchemaPreviewResponse,
    summary="Read YAML manifest → connect to Postgres → return raw schema (no Neo4j, no LLM)",
)
async def preview_schema(domain_id: str) -> SchemaPreviewResponse:
    """
    This endpoint shows exactly how the YAML and the database are connected.

    **Step 1 — Read the YAML**
    Opens `manifests/{domain_id}.yaml` and reads the `database_connection` block:

    ```yaml
    database_connection:
      env_vars:
        host:     CHINOOK_DB_HOST   # ← this is just a name, not a value
        port:     CHINOOK_DB_PORT
        database: CHINOOK_DB_NAME
        user:     CHINOOK_DB_USER
        password: CHINOOK_DB_PASSWORD
    ```

    **Step 2 — Resolve credentials from .env**
    Each value in `env_vars` is an env-var *name*. The extractor calls
    `os.environ.get("CHINOOK_DB_HOST")` etc. to get actual credentials at runtime.

    **Step 3 — Connect to Postgres and extract schema**
    Runs `information_schema` queries to pull every table, column, data type,
    primary key, unique constraint, foreign key, and `pg_description` comment.

    **Nothing is written anywhere.** This is read-only and has no side effects.
    Use it to verify the YAML → Postgres connection before running a full ingest.
    """
    # ── Step 1: Load the YAML manifest ───────────────────────────────────
    loader = get_manifest_loader()
    try:
        manifest = await loader.load(domain_id)
    except ManifestNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # ── Step 2 & 3: Use database_connection from YAML to query Postgres ──
    # manifest["database_connection"]["env_vars"] looks like:
    #   { "host": "CHINOOK_DB_HOST", "password": "CHINOOK_DB_PASSWORD", ... }
    # extract_schema() does os.environ.get("CHINOOK_DB_HOST") etc. internally.
    db_connection = manifest.get("database_connection", {})
    try:
        schema = await extract_schema(db_connection)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Cannot connect to Postgres for '{domain_id}'. "
                f"Check that env vars in manifests/{domain_id}.yaml → database_connection "
                f"are set in your .env file. Error: {exc}"
            ),
        ) from exc

    tables = schema["tables"]
    return SchemaPreviewResponse(
        domain=domain_id,
        table_count=len(tables),
        column_count=sum(len(t["columns"]) for t in tables),
        tables=tables,
    )


@router.post(
    "/ingest/{domain_id}",
    response_model=IngestResponse,
    summary="Load manifest + extract Postgres schema + enrich descriptions + store in Neo4j",
)
async def ingest_domain(domain_id: str) -> IngestResponse:
    """
    Single endpoint that runs the full ingestion pipeline for a domain:

    **Step 1 — Load manifest** (`manifests/{domain_id}.yaml`)
    Contains persona, business glossary, business rules, sensitivity rules,
    row limits, and few-shot Q→SQL examples.

    **Step 2 — Extract schema from Postgres**
    Connects to the database defined in the manifest's `database_connection`
    block. Reads credentials from env vars (never hardcoded). Extracts every
    table, column, data type, PK, FK relationship, and `pg_description` comment.

    **Step 3 — Merge live schema into manifest**
    Overwrites the manifest's static `tables:` block with the live columns
    so the knowledge graph always reflects the actual database.

    **Step 4 — Enrich blank descriptions via LLM**
    Calls Claude once per table that has missing descriptions. Tables and
    columns that already have `pg_description` comments are left unchanged.

    **Step 5 — Store in Neo4j AuraDB**
    Deletes any existing nodes for this domain (clean rebuild), batches all
    node descriptions into a single OpenAI embedding call, then writes
    TABLE/COLUMN nodes and HAS_COLUMN/REFERENCES/RELATED_TO edges to Neo4j
    using UNWIND for efficiency.

    Raises **404** if the manifest file does not exist.
    Raises **502** if the Postgres database is unreachable.
    Raises **500** if the Neo4j write fails.
    """

    t0 = time.monotonic()

    # ── Step 1: Load manifest ─────────────────────────────────────────────
    loader = get_manifest_loader()
    try:
        manifest = await loader.load(domain_id)
    except ManifestNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    logger.info("[%s] Step 1 ✓ Manifest loaded", domain_id)

    # ── Step 2: Extract live schema from Postgres ─────────────────────────
    db_connection = manifest.get("database_connection", {})
    try:
        schema = await extract_schema(db_connection)
    except Exception as exc:
        logger.error("[%s] Postgres extraction failed: %s", domain_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Cannot connect to Postgres for domain '{domain_id}'. "
                f"Check the env vars in manifests/{domain_id}.yaml → database_connection. "
                f"Error: {exc}"
            ),
        ) from exc

    tables     = schema["tables"]
    col_count  = sum(len(t["columns"]) for t in tables)
    fk_count   = sum(
        len(c.get("foreign_keys", []))
        for t in tables
        for c in t["columns"]
    )
    logger.info(
        "[%s] Step 2 ✓ Schema extracted — %d tables, %d columns, %d FK edges",
        domain_id, len(tables), col_count, fk_count,
    )

    # ── Step 3: Merge live schema into manifest ───────────────────────────
    # The manifest has a static `tables:` block written by hand.
    # Replace it with the live columns from Postgres so Neo4j always
    # reflects the real database structure.
    manifest["tables"] = tables
    logger.info("[%s] Step 3 ✓ Live schema merged into manifest", domain_id)

    # ── Step 4: Enrich blank descriptions via LLM ─────────────────────────
    blank_tables = sum(1 for t in tables if not t.get("description", "").strip())
    blank_cols   = sum(
        1 for t in tables
        for c in t["columns"]
        if not c.get("description", "").strip()
    )
    if blank_tables > 0 or blank_cols > 0:
        logger.info(
            "[%s] Step 4 — LLM enriching %d table + %d column blank descriptions …",
            domain_id, blank_tables, blank_cols,
        )
        try:
            schema = await enrich_descriptions(schema)
            logger.info("[%s] Step 4 ✓ Descriptions enriched", domain_id)
        except Exception as exc:
            # Non-fatal: proceed with blanks rather than fail the entire ingest
            logger.warning(
                "[%s] Step 4 ⚠ LLM enrichment failed (%s) — proceeding with blanks",
                domain_id, exc,
            )
    else:
        logger.info("[%s] Step 4 ✓ All descriptions present — LLM skipped", domain_id)

    # ── Step 5: Write to Neo4j AuraDB ─────────────────────────────────────
    glossary    = manifest.get("business_glossary", {}) or {}
    sensitivity = manifest.get("sensitivity", []) or []
    try:
        kg = _get_kg()
        embeddings_count = await kg.build(
            schema, domain_id, glossary=glossary, sensitivity=sensitivity
        )

        # Mark domain as built in the RAG service cache
        if domain_id not in _build_locks:
            _build_locks[domain_id] = asyncio.Lock()
        async with _build_locks[domain_id]:
            _domain_built.add(domain_id)

        duration_ms = int((time.monotonic() - t0) * 1000)
        _write_audit_log(
            domain_id,
            tables_ingested=len(tables),
            columns_ingested=col_count,
            embeddings_generated=embeddings_count,
            duration_ms=duration_ms,
        )
        logger.info("[%s] Step 5 ✓ Knowledge graph stored in Neo4j", domain_id)

    except RuntimeError as exc:
        # _get_kg() raises RuntimeError if init_driver() was never called
        raise HTTPException(
            status_code=500,
            detail="Neo4j driver not initialised. Is the server running? Check startup logs.",
        ) from exc
    except Exception as exc:
        logger.error("[%s] Neo4j write failed: %s", domain_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write knowledge graph to Neo4j for '{domain_id}'. Error: {exc}",
        ) from exc

    return IngestResponse(
        domain=domain_id,
        status="ingested",
        table_count=len(tables),
        column_count=col_count,
        fk_count=fk_count,
        embeddings_generated=embeddings_count,
        duration_ms=duration_ms,
        message=(
            f"Domain '{domain_id}' fully ingested: "
            f"{len(tables)} tables, {col_count} columns, {fk_count} FK edges, "
            f"{len(glossary)} business terms stored in Neo4j with {embeddings_count} embeddings "
            f"in {duration_ms} ms."
        ),
    )


@router.delete(
    "/ingest/{domain_id}",
    response_model=IngestResponse,
    summary="Delete domain knowledge graph from Neo4j (next query rebuilds automatically)",
)
async def delete_domain(domain_id: str) -> IngestResponse:
    """
    Wipe all Neo4j nodes for *domain_id*.

    Use this after a database schema migration or after editing the manifest.
    The next call to `POST /api/v1/query` for this domain will automatically
    re-run the full ingestion pipeline.
    """
    loader = get_manifest_loader()
    try:
        await loader.load(domain_id)
    except ManifestNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        await invalidate_cache(domain_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete Neo4j graph for '{domain_id}'. Error: {exc}",
        ) from exc

    return IngestResponse(
        domain=domain_id,
        status="deleted",
        table_count=0,
        column_count=0,
        fk_count=0,
        message=f"Knowledge graph for '{domain_id}' deleted. Next query will rebuild it.",
    )



# ── Step 1: RETRIEVE — preview schema from Postgres ──────────────────────────

@router.get(
    "/ingest/{domain_id}/schema",
    response_model=SchemaPreviewResponse,
    summary="Step 1 — Retrieve raw schema from Postgres (preview, no Neo4j write)",
)
async def preview_schema(domain_id: str) -> SchemaPreviewResponse:
    """
    Connect to the domain's Postgres database, run ``information_schema``
    introspection, and return the extracted schema.

    **Nothing is written to Neo4j** — this is a read-only preview step so
    you can validate the schema before ingestion.

    Raises 404 if the domain manifest does not exist.
    Raises 502 if the Postgres database is unreachable.
    """
    loader = get_manifest_loader()
    try:
        manifest = await loader.load(domain_id)
    except ManifestNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db_connection = manifest.get("database_connection", {})

    try:
        schema = await extract_schema(db_connection)
    except Exception as exc:
        logger.error("Schema extraction failed for domain '%s': %s", domain_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Could not connect to the Postgres database for domain '{domain_id}'. "
                f"Check that the env vars in the manifest database_connection block are set. "
                f"Error: {exc}"
            ),
        ) from exc

    tables = schema.get("tables", [])
    table_summaries = [
        {
            "name":           t["name"],
            "schema":         t.get("schema", "public"),
            "description":    t.get("description", ""),
            "estimated_rows": t.get("estimated_rows", 0),
            "column_count":   len(t.get("columns", [])),
            "columns": [
                {
                    "name":        c["name"],
                    "type":        c["type"],
                    "primary_key": c.get("primary_key", False),
                    "nullable":    c.get("nullable", True),
                    "description": c.get("description", ""),
                    "foreign_keys": c.get("foreign_keys", []),
                }
                for c in t.get("columns", [])
            ],
        }
        for t in tables
    ]

    return SchemaPreviewResponse(
        domain=domain_id,
        table_count=len(tables),
        column_count=sum(len(t.get("columns", [])) for t in tables),
        tables=table_summaries,
    )


# ── Step 2: STORE — retrieve schema then write to Neo4j ──────────────────────

@router.post(
    "/ingest/{domain_id}",
    response_model=IngestResponse,
    summary="Step 2 — Retrieve schema from Postgres and store in Neo4j KG",
)
async def ingest_domain(domain_id: str) -> IngestResponse:
    """
    **Two-step pipeline (explicit):**

    1. **Retrieve** — connect to Postgres and extract tables, columns, PKs,
       FKs and ``pg_description`` comments via ``information_schema``.

    2. **Store** — embed all node descriptions in one batched OpenAI call,
       then write TABLE/COLUMN nodes + relationship edges to Neo4j AuraDB
       using the 5.x vector index for future ANN retrieval.

    The operation is idempotent: existing nodes for the domain are deleted
    before the new ones are written (clean rebuild).

    Raises 404 if the manifest does not exist.
    Raises 502 if Postgres is unreachable.
    Raises 500 if the Neo4j write fails.
    """
    loader = get_manifest_loader()
    try:
        manifest = await loader.load(domain_id)
    except ManifestNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    db_connection = manifest.get("database_connection", {})

    # ── Step 1: Retrieve schema from Postgres ────────────────────────────
    logger.info("[INGEST] Step 1 — Retrieving schema for domain '%s' from Postgres", domain_id)
    try:
        schema = await extract_schema(db_connection)
    except Exception as exc:
        logger.error("[INGEST] Schema extraction failed for '%s': %s", domain_id, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Could not retrieve schema from Postgres for domain '{domain_id}'. "
                f"Error: {exc}"
            ),
        ) from exc

    tables      = schema.get("tables", [])
    table_count = len(tables)
    col_count   = sum(len(t.get("columns", [])) for t in tables)
    fk_count    = sum(
        len(c.get("foreign_keys", []))
        for t in tables
        for c in t.get("columns", [])
    )
    logger.info(
        "[INGEST] Step 1 complete — %d tables, %d columns, %d FK edges extracted",
        table_count, col_count, fk_count,
    )
    logger.info("[INGEST] Schema for domain '%s': %s", domain_id, schema)
    # ── Step 2: Write to Neo4j ────────────────────────────────────────────
    logger.info("[INGEST] Step 2 — Writing schema for domain '%s' to Neo4j", domain_id)
    try:
        kg = _get_kg()
        await kg.build(schema, domain_id)

        # Mark domain as freshly built in the RAG service
        if domain_id in _build_locks:
            async with _build_locks[domain_id]:
                _domain_built.add(domain_id)
        else:
            _domain_built.add(domain_id)

    except Exception as exc:
        logger.error("[INGEST] Neo4j write failed for '%s': %s", domain_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write schema to Neo4j for domain '{domain_id}'. Error: {exc}",
        ) from exc

    logger.info("[INGEST] Step 2 complete — domain '%s' KG ready in Neo4j", domain_id)

    return IngestResponse(
        domain=domain_id,
        status="ingested",
        table_count=table_count,
        column_count=col_count,
        fk_count=fk_count,
        message=(
            f"Schema for domain '{domain_id}' successfully retrieved from Postgres "
            f"and stored in Neo4j ({table_count} tables, {col_count} columns, "
            f"{fk_count} FK edges)."
        ),
    )


# ── Invalidate (force rebuild) ────────────────────────────────────────────────

@router.delete(
    "/ingest/{domain_id}",
    response_model=IngestResponse,
    summary="Delete domain KG from Neo4j — next query triggers a full rebuild",
)
async def invalidate_domain(domain_id: str) -> IngestResponse:
    """
    Delete all Neo4j nodes for *domain_id* and clear the in-process build
    flag.  The next query to this domain will trigger a fresh
    retrieve-from-Postgres → store-to-Neo4j cycle automatically.

    Use this after database schema migrations or manifest updates.
    """
    loader = get_manifest_loader()
    try:
        await loader.load(domain_id)   # confirm domain exists
    except ManifestNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        await invalidate_cache(domain_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to invalidate KG for domain '{domain_id}'. Error: {exc}",
        ) from exc

    return IngestResponse(
        domain=domain_id,
        status="invalidated",
        table_count=0,
        column_count=0,
        fk_count=0,
        message=(
            f"Knowledge graph for domain '{domain_id}' deleted from Neo4j. "
            "Next query will trigger a full rebuild."
        ),
    )

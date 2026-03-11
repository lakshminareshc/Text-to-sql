"""
scripts/ingest_schema.py
────────────────────────
Standalone ingestion script — run this BEFORE starting the API server.

Pipeline (explicit, sequential):

  Step 1  RETRIEVE
          Connect to the domain's Postgres database and extract the full
          schema via information_schema: tables, columns, PKs, UKs, FKs,
          and pg_description comments.

  Step 2  ENRICH
          For every table/column that has NO pg_description comment, call
          the LLM to generate a concise plain-English description.
          Tables and columns that already have comments are left unchanged.

  Step 3  STORE
          Open the Neo4j AuraDB driver, ensure the vector index exists,
          embed all node descriptions in a single OpenAI batch call, then
          write TABLE/COLUMN nodes + HAS_COLUMN/REFERENCES/RELATED_TO
          edges to Neo4j.

Usage
-----
  # Ingest a single domain
  python -m scripts.ingest_schema --domain sales

  # Ingest all domains found in the manifest directory
  python -m scripts.ingest_schema --all

  # Preview schema without writing to Neo4j (dry-run)
  python -m scripts.ingest_schema --domain sales --dry-run

  # Force rebuild (delete existing nodes first)
  python -m scripts.ingest_schema --domain sales --rebuild
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# ── make sure the project root is on sys.path ─────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from app.config import get_settings
from app.manifests.manifest_loader import get_manifest_loader
from app.services.schema_extractor import enrich_descriptions, extract_schema
from app.services.schema_knowledge_graph import SchemaKnowledgeGraph
from neo4j import AsyncGraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def ingest_domain(
    domain_id: str,
    *,
    dry_run: bool = False,
    rebuild: bool = False,
) -> None:
    settings = get_settings()
    loader   = get_manifest_loader()

    logger.info("══════════════════════════════════════════════")
    logger.info("Domain : %s", domain_id)
    logger.info("DryRun : %s  |  Rebuild : %s", dry_run, rebuild)
    logger.info("══════════════════════════════════════════════")

    # ── Load manifest ─────────────────────────────────────────────────────
    manifest = await loader.load(domain_id)
    db_connection = manifest.get("database_connection", {})

    # ────────────────────────────────────────────────────────────────────
    # STEP 1 — RETRIEVE schema from Postgres
    # ────────────────────────────────────────────────────────────────────
    logger.info("Step 1 ── Retrieving schema from Postgres …")
    try:
        schema = await extract_schema(db_connection)
    except Exception as exc:
        logger.error("  ✗ Could not connect to Postgres: %s", exc)
        logger.error("  Check that the env vars in the manifest are set in .env")
        raise SystemExit(1)

    tables     = schema["tables"]
    col_count  = sum(len(t["columns"]) for t in tables)
    fk_count   = sum(len(c.get("foreign_keys", [])) for t in tables for c in t["columns"])
    logger.info("  ✓ %d tables  |  %d columns  |  %d FK edges", len(tables), col_count, fk_count)

    # Print schema summary
    for t in tables:
        desc_flag  = "✓" if t.get("description") else "✗"
        col_flags  = sum(1 for c in t["columns"] if c.get("description"))
        logger.info(
            "    [%s desc] %-30s  %d cols  (%d/%d with desc)",
            desc_flag, t["name"], len(t["columns"]), col_flags, len(t["columns"]),
        )

    # ────────────────────────────────────────────────────────────────────
    # STEP 2 — ENRICH descriptions via LLM (blanks only)
    # ────────────────────────────────────────────────────────────────────
    blank_tables = sum(1 for t in tables if not t.get("description", "").strip())
    blank_cols   = sum(
        1 for t in tables for c in t["columns"]
        if not c.get("description", "").strip()
    )
    logger.info(
        "Step 2 ── Enriching descriptions (LLM fills %d table + %d column blanks) …",
        blank_tables, blank_cols,
    )

    if blank_tables == 0 and blank_cols == 0:
        logger.info("  ✓ All descriptions already present — skipping LLM enrichment")
    else:
        try:
            schema = await enrich_descriptions(schema)
            logger.info("  ✓ Descriptions enriched")
        except Exception as exc:
            logger.warning("  ⚠ LLM enrichment failed (%s) — proceeding with blanks", exc)

    # ── Print final description coverage ──────────────────────────────
    if not dry_run:
        for t in schema["tables"]:
            col_flags = sum(1 for c in t["columns"] if c.get("description"))
            logger.info(
                "    %-30s  table_desc=%r  cols=%d/%d described",
                t["name"],
                (t.get("description") or "")[:60],
                col_flags,
                len(t["columns"]),
            )

    if dry_run:
        logger.info("DRY RUN — skipping Neo4j write. Final schema JSON:")
        print(json.dumps(schema, indent=2, default=str))
        return

    # ────────────────────────────────────────────────────────────────────
    # STEP 3 — STORE in Neo4j AuraDB
    # ────────────────────────────────────────────────────────────────────
    logger.info("Step 3 ── Connecting to Neo4j AuraDB …")
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=10,
    )
    try:
        await driver.verify_connectivity()
        logger.info("  ✓ Connected to %s", settings.neo4j_uri)

        kg = SchemaKnowledgeGraph(driver, settings.neo4j_database)

        # Ensure vector index exists (idempotent)
        logger.info("  Creating vector index '%s' if not exists …", settings.neo4j_vector_index_name)
        await kg.ensure_vector_index()
        logger.info("  ✓ Vector index ready")

        if rebuild:
            logger.info("  Deleting existing nodes for domain '%s' …", domain_id)
            await kg.delete_domain(domain_id)
            logger.info("  ✓ Existing nodes removed")

        logger.info("  Writing %d tables + %d columns to Neo4j (embedding in 1 batch) …",
                    len(schema["tables"]), col_count)
        await kg.build(schema, domain_id)
        logger.info("  ✓ Knowledge graph stored in Neo4j")

    finally:
        await driver.close()
        logger.info("  Neo4j driver closed")

    logger.info("══════════════════════════════════════════════")
    logger.info("✓ Ingestion complete for domain '%s'", domain_id)
    logger.info("══════════════════════════════════════════════")


async def ingest_all(*, dry_run: bool = False, rebuild: bool = False) -> None:
    settings     = get_settings()
    manifest_dir = Path(settings.manifest_local_path)
    domains      = [p.stem for p in manifest_dir.glob("*.yaml")]

    if not domains:
        logger.error("No manifest YAML files found in '%s'", manifest_dir)
        raise SystemExit(1)

    logger.info("Found %d domains: %s", len(domains), domains)
    for domain_id in domains:
        await ingest_domain(domain_id, dry_run=dry_run, rebuild=rebuild)


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve Postgres schema → enrich descriptions via LLM → store in Neo4j KG"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--domain", metavar="DOMAIN_ID",
                       help="Domain to ingest (e.g. sales)")
    group.add_argument("--all", action="store_true",
                       help="Ingest all domains found in the manifest directory")

    parser.add_argument("--dry-run", action="store_true",
                        help="Preview schema + descriptions without writing to Neo4j")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing KG nodes before writing (full rebuild)")

    args = parser.parse_args()

    if args.domain:
        asyncio.run(ingest_domain(args.domain, dry_run=args.dry_run, rebuild=args.rebuild))
    else:
        asyncio.run(ingest_all(dry_run=args.dry_run, rebuild=args.rebuild))


if __name__ == "__main__":
    main()

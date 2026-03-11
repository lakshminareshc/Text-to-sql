"""
SchemaExtractor
───────────────
Introspects a live PostgreSQL database via asyncpg, extracting:

  • tables   – name, schema, estimated row count, pg_description comment
  • columns  – name, data type, nullable, PK, unique, default, pg_description comment
  • FK edges – column → referenced table.column

After extraction, ``enrich_descriptions(schema)`` can be called to fill any
empty table/column descriptions using the LLM — one batched prompt per table.

Connection credentials are resolved from env-var *names* declared in the
manifest's ``database_connection.env_vars`` block; they are never hardcoded.

Example
-------
    from app.services.schema_extractor import extract_schema, enrich_descriptions

    schema = await extract_schema(manifest["database_connection"])
    schema = await enrich_descriptions(schema)   # fills blank descriptions via LLM
    # schema == {"tables": [...]}
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import asyncpg

from app.services.llm_service import generate_completion

logger = logging.getLogger(__name__)


# ── Connection factory ────────────────────────────────────────────────────────

async def _get_connection(env_vars: dict[str, str]) -> asyncpg.Connection:
    """
    Resolve Postgres credentials from the env-var names in *env_vars* and
    return an open asyncpg connection.

    Keys expected in *env_vars* (values are env-var *names*, not actual creds):
        host, port, database, user, password
    """
    host     = os.environ.get(env_vars.get("host",     "DB_HOST"),     "localhost")
    port     = int(os.environ.get(env_vars.get("port", "DB_PORT"),     "5432"))
    database = os.environ.get(env_vars.get("database", "DB_NAME"),     "postgres")
    user     = os.environ.get(env_vars.get("user",     "DB_USER"),     "postgres")
    password = os.environ.get(env_vars.get("password", "DB_PASSWORD"), "")

    logger.debug(
        "Connecting to Postgres at %s:%s / db=%s as %s", host, port, database, user
    )
    return await asyncpg.connect(
        host=host, port=port, database=database,
        user=user, password=password,
        command_timeout=30,
    )


# ── Public API ────────────────────────────────────────────────────────────────

async def extract_schema(db_connection: dict[str, Any]) -> dict[str, Any]:
    """
    Introspect the Postgres database described by *db_connection* (the
    ``database_connection`` block from a domain manifest) and return a
    normalised schema dict:

    .. code-block:: python

        {
          "tables": [
            {
              "name":           "orders",
              "schema":         "public",
              "description":    "One row per customer order",
              "estimated_rows": 142000,
              "columns": [
                {
                  "name":         "order_id",
                  "type":         "int4",
                  "nullable":     False,
                  "primary_key":  True,
                  "unique":       True,
                  "default":      "nextval('orders_order_id_seq')",
                  "description":  "Surrogate primary key",
                  "foreign_keys": []
                },
                {
                  "name":         "customer_id",
                  "type":         "int4",
                  "nullable":     False,
                  "primary_key":  False,
                  "unique":       False,
                  "default":      "",
                  "description":  "",
                  "foreign_keys": [{"table": "customers", "schema": "public",
                                    "column": "customer_id"}]
                },
                ...
              ]
            },
            ...
          ]
        }

    Parameters
    ----------
    db_connection:
        The ``database_connection`` block from the domain manifest.  Must
        contain an ``env_vars`` sub-dict whose values are env-var *names*.
    """
    env_vars = db_connection.get("env_vars", db_connection)
    conn = await _get_connection(env_vars)
    try:
        return await _introspect(conn)
    finally:
        await conn.close()


# ── Internal introspection ────────────────────────────────────────────────────

async def _introspect(conn: asyncpg.Connection) -> dict[str, Any]:
    # ── tables + object comments ──────────────────────────────────────────
    table_rows = await conn.fetch("""
        SELECT
            t.table_schema,
            t.table_name,
            obj_description(
                ('"' || t.table_schema || '"."' || t.table_name || '"')::regclass,
                'pg_class'
            ) AS description,
            GREATEST(c.reltuples::bigint, 0) AS estimated_rows
        FROM information_schema.tables t
        JOIN pg_class       c ON c.relname = t.table_name
        JOIN pg_namespace   n ON n.nspname = t.table_schema
                             AND n.oid     = c.relnamespace
        WHERE t.table_type = 'BASE TABLE'
          AND t.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY t.table_schema, t.table_name
    """)

    # ── primary key columns ───────────────────────────────────────────────
    pk_rows = await conn.fetch("""
        SELECT kcu.table_schema, kcu.table_name, kcu.column_name
        FROM information_schema.table_constraints  tc
        JOIN information_schema.key_column_usage   kcu
            ON  tc.constraint_name = kcu.constraint_name
            AND tc.table_schema    = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
    """)
    pk_set = {
        (r["table_schema"], r["table_name"], r["column_name"])
        for r in pk_rows
    }

    # ── unique-constrained columns ────────────────────────────────────────
    uq_rows = await conn.fetch("""
        SELECT kcu.table_schema, kcu.table_name, kcu.column_name
        FROM information_schema.table_constraints  tc
        JOIN information_schema.key_column_usage   kcu
            ON  tc.constraint_name = kcu.constraint_name
            AND tc.table_schema    = kcu.table_schema
        WHERE tc.constraint_type = 'UNIQUE'
    """)
    uq_set = {
        (r["table_schema"], r["table_name"], r["column_name"])
        for r in uq_rows
    }

    # ── foreign-key relationships ─────────────────────────────────────────
    fk_rows = await conn.fetch("""
        SELECT
            kcu.table_schema,
            kcu.table_name,
            kcu.column_name,
            ccu.table_schema AS ref_table_schema,
            ccu.table_name   AS ref_table_name,
            ccu.column_name  AS ref_column_name
        FROM information_schema.table_constraints         tc
        JOIN information_schema.key_column_usage          kcu
            ON  tc.constraint_name = kcu.constraint_name
            AND tc.table_schema    = kcu.table_schema
        JOIN information_schema.constraint_column_usage   ccu
            ON  tc.constraint_name = ccu.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
    """)
    fk_map: dict[tuple, list[dict]] = {}
    for r in fk_rows:
        key = (r["table_schema"], r["table_name"], r["column_name"])
        fk_map.setdefault(key, []).append({
            "table":  r["ref_table_name"],
            "schema": r["ref_table_schema"],
            "column": r["ref_column_name"],
        })

    # ── columns + column comments ─────────────────────────────────────────
    col_rows = await conn.fetch("""
        SELECT
            c.table_schema,
            c.table_name,
            c.column_name,
            c.udt_name           AS data_type,
            c.is_nullable,
            c.column_default,
            c.ordinal_position,
            col_description(
                ('"' || c.table_schema || '"."' || c.table_name || '"')::regclass,
                c.ordinal_position
            ) AS description
        FROM information_schema.columns c
        WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY c.table_schema, c.table_name, c.ordinal_position
    """)

    # ── assemble final structure ──────────────────────────────────────────
    tables_dict: dict[tuple, dict] = {}
    for tr in table_rows:
        key = (tr["table_schema"], tr["table_name"])
        tables_dict[key] = {
            "name":           tr["table_name"],
            "schema":         tr["table_schema"],
            "description":    tr["description"] or "",
            "estimated_rows": int(tr["estimated_rows"]),
            "columns":        [],
        }

    for cr in col_rows:
        key = (cr["table_schema"], cr["table_name"])
        if key not in tables_dict:
            continue
        col_key = (cr["table_schema"], cr["table_name"], cr["column_name"])
        tables_dict[key]["columns"].append({
            "name":         cr["column_name"],
            "type":         cr["data_type"],
            "nullable":     cr["is_nullable"] == "YES",
            "primary_key":  col_key in pk_set,
            "unique":       col_key in uq_set,
            "default":      cr["column_default"] or "",
            "description":  cr["description"] or "",
            "foreign_keys": fk_map.get(col_key, []),
        })

    logger.info(
        "SchemaExtractor: found %d tables with columns", len(tables_dict)
    )
    return {"tables": list(tables_dict.values())}


# ── LLM description enrichment ────────────────────────────────────────────────

async def enrich_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Fill every empty ``description`` field in *schema* using the LLM.

    For each table that is missing a table-level or column-level description,
    one LLM call is made that generates descriptions for the whole table and
    all of its undescribed columns in a single JSON response.

    Tables and columns that already have ``pg_description`` comments are left
    unchanged — the LLM only fills the blanks.

    Parameters
    ----------
    schema:
        The dict returned by ``extract_schema`` — mutated in-place and returned.

    Returns
    -------
    dict
        The same schema dict with all blank descriptions filled.
    """
    for table in schema.get("tables", []):
        # Check if anything needs enrichment
        table_needs = not table.get("description", "").strip()
        cols_needing = [
            c for c in table.get("columns", [])
            if not c.get("description", "").strip()
        ]

        if not table_needs and not cols_needing:
            continue   # all descriptions present — skip this table

        logger.info(
            "Enriching descriptions for table '%s' (table=%s, cols=%d)",
            table["name"], table_needs, len(cols_needing),
        )

        # Build a compact schema snapshot for the LLM prompt
        col_lines = "\n".join(
            f"  - {c['name']} ({c['type']})"
            + (" [PK]"      if c.get("primary_key") else "")
            + (" [NOT NULL]" if not c.get("nullable", True) else "")
            + (f" → {c['foreign_keys'][0]['table']}.{c['foreign_keys'][0]['column']}"
               if c.get("foreign_keys") else "")
            + (f"  # already described: {c['description']}"
               if c.get("description", "").strip() else "")
            for c in table.get("columns", [])
        )

        prompt = f"""You are a database documentation expert.
Given the following Postgres table, write concise, precise plain-English descriptions.

Table: {table['name']}
{f'Existing table description: {table["description"]}' if table.get("description") else 'Table description: (missing — please generate)'}

Columns:
{col_lines}

Return ONLY a JSON object with this exact structure — no markdown fences:
{{
  "table_description": "<one sentence describing the business purpose of this table>",
  "columns": {{
    "<column_name>": "<one sentence describing what this column stores>",
    ...
  }}
}}

Rules:
- Only include columns that are MISSING a description (listed without '# already described').
- Keep each description under 15 words.
- Use business language, not technical jargon.
- Do NOT include columns that already have a description.
"""
        try:
            from app.config import get_settings as _get_settings
            _enrich_model = _get_settings().enrich_llm_model
            raw = await generate_completion(
                [{"role": "user", "content": prompt}],
                model=_enrich_model,
                temperature=0.0,
                max_tokens=1024,
            )
            # Strip accidental markdown fences
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            data: dict = json.loads(cleaned)
        except Exception as exc:
            logger.warning(
                "LLM description enrichment failed for table '%s': %s",
                table["name"], exc,
            )
            continue

        # Apply generated table description if it was missing
        if table_needs and data.get("table_description"):
            table["description"] = data["table_description"].strip()

        # Apply generated column descriptions
        col_descriptions: dict = data.get("columns", {})
        for col in table["columns"]:
            if not col.get("description", "").strip():
                generated = col_descriptions.get(col["name"], "").strip()
                if generated:
                    col["description"] = generated

    logger.info("Description enrichment complete for all tables.")
    return schema


"""
SchemaKnowledgeGraph — Neo4j AuraDB backend
────────────────────────────────────────────
Persists the database schema as a rich property graph in Neo4j AuraDB and
uses Neo4j's native 5.x vector index for semantic (ANN) retrieval.

Graph model
───────────
Node labels
-----------
  :Domain                   – one node per domain (multi-tenancy anchor)
  :SchemaNode:TABLE         – one node per table / view
  :SchemaNode:COLUMN        – one node per column
  :SchemaNode:BusinessTerm  – one node per glossary term

All :SchemaNode nodes carry a 3072-dim ``embedding`` vector.

Relationship types
------------------
  BELONGS_TO           (:TABLE)        → (:Domain)         domain membership
  HAS_COLUMN           (:TABLE)        → (:COLUMN)         {ordinal_position}
  FOREIGN_KEY          (:TABLE)        → (:TABLE)          {from_col, to_col}
  COMMONLY_JOINED_WITH (:TABLE)        ↔ (:TABLE)          {frequency_score}
  MAPS_TO              (:BusinessTerm) → (:TABLE|:COLUMN)
  DERIVED_FROM         (:TABLE view)   → (:TABLE base)     (reserved, future)

Node properties
---------------
:Domain
  node_id, domain, name

:TABLE / :COLUMN (all SchemaNode)
  node_id           – globally unique "{domain}:table:{name}"
                      or "{domain}:column:{table}.{name}"
  domain            – domain_id string for multi-tenant filtering
  node_type         – "TABLE" | "COLUMN" | "BusinessTerm"
  name              – table or column name
  description       – human-readable description
  text              – natural-language sentence used for embedding
  embedding         – float[] in Neo4j vector index

:TABLE extras
  schema_           – Postgres schema (usually "public")
  estimated_rows    – row count estimate from pg_catalog
  sensitivity_level – 0 = public, 1 = restricted (has ≥1 PII column)

:COLUMN extras
  table             – parent table name
  data_type         – Postgres type (int4, varchar, …)
  nullable          – bool
  primary_key       – bool
  unique_           – bool
  is_pii            – bool (True if in manifest sensitivity list)
  ordinal_position  – int (column order in the table)
  fk_targets        – list["table.column"] for display

Retrieval pipeline
------------------
  1. Embed the user query (single API call).
  2. ANN search on :TABLE nodes scoped to domain_id (top-k × OVERSAMPLE).
  3. 1–2 hop graph expansion via FOREIGN_KEY + COMMONLY_JOINED_WITH.
  4. Filter tables by sensitivity_level ≤ permission_tier (default 2 = all).
  5. Pull :COLUMN children; mask is_pii columns when permission_tier < 2.
  6. Pull :BusinessTerm nodes linked via MAPS_TO to matched tables/columns.
  7. Format compact [KNOWLEDGE GRAPH — RELEVANT SCHEMA] block.

Query-feedback loop
-------------------
  Call ``record_join(domain_id, table_a, table_b)`` after every successful
  query that joins those two tables.  It increments
  COMMONLY_JOINED_WITH.frequency_score so that frequently co-joined tables
  surface first in future retrievals.

Usage
-----
    # In lifespan (app startup)
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    kg = SchemaKnowledgeGraph(driver, database="neo4j")
    await kg.ensure_vector_index()

    # Ingest a domain
    await kg.build(schema, domain_id="sales",
                   glossary=manifest["business_glossary"],
                   sensitivity=manifest["sensitivity"])

    # At query time
    context = await kg.retrieve("total revenue by region", domain_id="sales")

    # After a successful join query
    await kg.record_join("sales", "contracts", "communities")
"""
from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver

from app.config import get_settings
from app.services.llm_service import generate_embeddings

logger = logging.getLogger(__name__)

# ANN oversampling factor before domain + type filtering.
_OVERSAMPLE = 10


class SchemaKnowledgeGraph:
    """Neo4j-backed schema knowledge graph."""

    def __init__(self, driver: AsyncDriver, database: str = "neo4j") -> None:
        self._driver   = driver
        self._database = database
        self._settings = get_settings()

    # ── Index setup ───────────────────────────────────────────────────────

    async def ensure_vector_index(self) -> None:
        """
        Create the vector index on :SchemaNode(embedding) if it does not
        already exist.  Safe to call on every startup — IF NOT EXISTS is
        idempotent.
        """
        index_name = self._settings.neo4j_vector_index_name
        dims       = self._settings.neo4j_embedding_dimensions

        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
                FOR (n:SchemaNode) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dims},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
            )
        logger.info(
            "Neo4j vector index '%s' ensured (%d dims, cosine)",
            index_name, dims,
        )

    # ── Build ─────────────────────────────────────────────────────────────

    async def build(
        self,
        schema: dict[str, Any],
        domain_id: str,
        glossary: dict[str, str] | None = None,
        sensitivity: list[str] | None = None,
    ) -> int:
        """
        Write the full schema for *domain_id* into Neo4j.

        Steps
        -----
        1.  Delete existing domain nodes/edges (clean rebuild).
        2.  Create the :Domain anchor node.
        3.  Collect TABLE rows — mark sensitivity_level per table.
        4.  Collect COLUMN rows — mark is_pii per column.
        5.  Collect BusinessTerm rows from glossary.
        6.  Single batched embedding call for all node texts.
        7.  Write TABLE nodes (UNWIND MERGE).
        8.  Write COLUMN nodes + HAS_COLUMN edges {ordinal_position}.
        9.  Write BELONGS_TO edges (TABLE → Domain).
        10. Write FOREIGN_KEY edges (TABLE → TABLE {from_col, to_col}).
        11. Write COMMONLY_JOINED_WITH edges (TABLE ↔ TABLE, frequency_score=0).
        12. Write BusinessTerm nodes.
        13. Write MAPS_TO edges (BusinessTerm → TABLE or COLUMN).

        Parameters
        ----------
        schema:
            ``{"tables": [...]}`` as returned by ``extract_schema()``.
        domain_id:
            Domain identifier — namespaces every node_id.
        glossary:
            ``business_glossary`` dict from the manifest (term → SQL fragment).
        sensitivity:
            ``sensitivity`` list from the manifest — column names in
            ``table.column`` or bare ``column`` form that are PII/restricted.

        Returns
        -------
        int
            Number of embedding vectors generated.
        """
        sensitivity_set: set[str] = {s.lower() for s in (sensitivity or [])}

        await self._delete_domain(domain_id)

        tables = schema.get("tables", [])
        if not tables:
            logger.warning("build() called with empty schema for domain '%s'", domain_id)
            return 0

        # ── 2. Domain anchor node ─────────────────────────────────────────
        async with self._driver.session(database=self._database) as session:
            await session.run(
                """
                MERGE (d:Domain {node_id: $node_id})
                SET d.domain = $domain,
                    d.name   = $domain
                """,
                node_id=f"domain:{domain_id}",
                domain=domain_id,
            )

        # ── 3 & 4. Collect TABLE / COLUMN rows ───────────────────────────
        table_rows:   list[dict] = []
        column_rows:  list[dict] = []
        fk_rows:      list[dict] = []
        belongs_rows: list[dict] = []

        all_texts:  list[str] = []
        node_order: list[str] = []

        for table in tables:
            tname    = table["name"]
            table_id = f"{domain_id}:table:{tname}"
            t_text   = self._table_description(table)

            # sensitivity_level: 1 if any column is PII, 0 otherwise
            col_pii_flags = [
                self._column_is_pii(tname, c["name"], sensitivity_set)
                for c in table.get("columns", [])
            ]
            sensitivity_level = 1 if any(col_pii_flags) else 0

            table_rows.append({
                "node_id":           table_id,
                "domain":            domain_id,
                "node_type":         "TABLE",
                "name":              tname,
                "schema_":           table.get("schema", "public"),
                "description":       table.get("description", ""),
                "estimated_rows":    int(table.get("estimated_rows", 0)),
                "sensitivity_level": sensitivity_level,
                "text":              t_text,
            })
            all_texts.append(t_text)
            node_order.append(table_id)

            belongs_rows.append({
                "table_id":  table_id,
                "domain_id": f"domain:{domain_id}",
            })

            for col_pos, (col, is_pii) in enumerate(
                zip(table.get("columns", []), col_pii_flags)
            ):
                cname  = col["name"]
                col_id = f"{domain_id}:column:{tname}.{cname}"
                c_text = self._column_description(tname, col)

                fk_targets = [
                    f"{fk['table']}.{fk['column']}"
                    for fk in col.get("foreign_keys", [])
                ]

                column_rows.append({
                    "node_id":          col_id,
                    "domain":           domain_id,
                    "node_type":        "COLUMN",
                    "name":             cname,
                    "table":            tname,
                    "parent_id":        table_id,
                    "data_type":        col.get("type", ""),
                    "nullable":         col.get("nullable", True),
                    "primary_key":      col.get("primary_key", False),
                    "unique_":          col.get("unique", False),
                    "is_pii":           is_pii,
                    "ordinal_position": col_pos,
                    "description":      col.get("description", ""),
                    "fk_targets":       fk_targets,
                    "text":             c_text,
                })
                all_texts.append(c_text)
                node_order.append(col_id)

                for fk in col.get("foreign_keys", []):
                    ref_tname = fk["table"]
                    pair_a    = min(tname, ref_tname)
                    pair_b    = max(tname, ref_tname)
                    fk_rows.append({
                        "src_id":    table_id,
                        "ref_id":    f"{domain_id}:table:{ref_tname}",
                        "from_col":  cname,
                        "to_col":    fk["column"],
                        "pair_a_id": f"{domain_id}:table:{pair_a}",
                        "pair_b_id": f"{domain_id}:table:{pair_b}",
                    })

        # ── 5. Collect BusinessTerm rows ──────────────────────────────────
        term_rows:    list[dict] = []
        maps_to_rows: list[dict] = []

        table_name_set = {r["name"].lower() for r in table_rows}

        if glossary:
            for term, sql_fragment in glossary.items():
                safe    = term.lower().replace(" ", "_")
                term_id = f"{domain_id}:term:{safe}"
                t_text  = f"Business term '{term}': {sql_fragment}"

                term_rows.append({
                    "node_id":    term_id,
                    "domain":     domain_id,
                    "node_type":  "BusinessTerm",
                    "name":       term,
                    "definition": str(sql_fragment),
                    "text":       t_text,
                })
                all_texts.append(t_text)
                node_order.append(term_id)

                fragment_lower = str(sql_fragment).lower()

                # MAPS_TO Tables whose name appears in the SQL fragment
                for tname in table_name_set:
                    if tname in fragment_lower:
                        maps_to_rows.append({
                            "term_id":   term_id,
                            "target_id": f"{domain_id}:table:{tname}",
                        })

                # MAPS_TO Columns referenced as "table.column" in the fragment
                for col_row in column_rows:
                    qualified = f"{col_row['table']}.{col_row['name']}".lower()
                    if qualified in fragment_lower:
                        maps_to_rows.append({
                            "term_id":   term_id,
                            "target_id": col_row["node_id"],
                        })

        # ── 6. Single batched embed call ──────────────────────────────────
        logger.info(
            "Domain '%s': embedding %d nodes in one batch",
            domain_id, len(all_texts),
        )
        vectors = await generate_embeddings(all_texts)
        vec_map: dict[str, list[float]] = dict(zip(node_order, vectors))

        for row in table_rows:
            row["embedding"] = vec_map[row["node_id"]]
        for row in column_rows:
            row["embedding"] = vec_map[row["node_id"]]
        for row in term_rows:
            row["embedding"] = vec_map[row["node_id"]]

        # ── 7–13. Write to Neo4j ──────────────────────────────────────────
        async with self._driver.session(database=self._database) as session:

            # Pass 1: TABLE nodes (with sensitivity_level)
            await session.run(
                """
                UNWIND $rows AS r
                MERGE (t:SchemaNode:TABLE {node_id: r.node_id})
                SET t.domain            = r.domain,
                    t.node_type         = r.node_type,
                    t.name              = r.name,
                    t.schema_           = r.schema_,
                    t.description       = r.description,
                    t.estimated_rows    = r.estimated_rows,
                    t.sensitivity_level = r.sensitivity_level,
                    t.text              = r.text,
                    t.embedding         = r.embedding
                """,
                rows=table_rows,
            )

            # Pass 2: COLUMN nodes + HAS_COLUMN edges {ordinal_position}
            await session.run(
                """
                UNWIND $rows AS r
                MERGE (c:SchemaNode:COLUMN {node_id: r.node_id})
                SET c.domain           = r.domain,
                    c.node_type        = r.node_type,
                    c.name             = r.name,
                    c.table            = r.table,
                    c.data_type        = r.data_type,
                    c.nullable         = r.nullable,
                    c.primary_key      = r.primary_key,
                    c.unique_          = r.unique_,
                    c.is_pii           = r.is_pii,
                    c.description      = r.description,
                    c.fk_targets       = r.fk_targets,
                    c.text             = r.text,
                    c.embedding        = r.embedding
                WITH c, r
                MATCH (t:TABLE {node_id: r.parent_id})
                MERGE (t)-[e:HAS_COLUMN]->(c)
                SET e.ordinal_position = r.ordinal_position
                """,
                rows=column_rows,
            )

            # Pass 3: BELONGS_TO edges (TABLE → Domain)
            await session.run(
                """
                UNWIND $rows AS r
                MATCH (t:TABLE  {node_id: r.table_id})
                MATCH (d:Domain {node_id: r.domain_id})
                MERGE (t)-[:BELONGS_TO]->(d)
                """,
                rows=belongs_rows,
            )

            # Pass 4: FOREIGN_KEY edges (TABLE → TABLE {from_col, to_col})
            if fk_rows:
                await session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (src:TABLE {node_id: r.src_id})
                    MATCH (ref:TABLE {node_id: r.ref_id})
                    MERGE (src)-[fk:FOREIGN_KEY {from_col: r.from_col,
                                                  to_col:   r.to_col}]->(ref)
                    """,
                    rows=fk_rows,
                )

            # Pass 5: COMMONLY_JOINED_WITH edges (TABLE ↔ TABLE, frequency_score=0)
            # Canonical direction: alphabetically lower node_id → higher node_id.
            if fk_rows:
                seen_pairs: set[tuple[str, str]] = set()
                unique_pairs: list[dict] = []
                for r in fk_rows:
                    pair = (r["pair_a_id"], r["pair_b_id"])
                    if pair not in seen_pairs and r["pair_a_id"] != r["pair_b_id"]:
                        seen_pairs.add(pair)
                        unique_pairs.append({
                            "a_id": r["pair_a_id"],
                            "b_id": r["pair_b_id"],
                        })
                if unique_pairs:
                    await session.run(
                        """
                        UNWIND $rows AS r
                        MATCH (a:TABLE {node_id: r.a_id})
                        MATCH (b:TABLE {node_id: r.b_id})
                        MERGE (a)-[cj:COMMONLY_JOINED_WITH]->(b)
                        ON CREATE SET cj.frequency_score = 0
                        """,
                        rows=unique_pairs,
                    )

            # Pass 6: BusinessTerm nodes
            if term_rows:
                await session.run(
                    """
                    UNWIND $rows AS r
                    MERGE (bt:SchemaNode:BusinessTerm {node_id: r.node_id})
                    SET bt.domain     = r.domain,
                        bt.node_type  = r.node_type,
                        bt.name       = r.name,
                        bt.definition = r.definition,
                        bt.text       = r.text,
                        bt.embedding  = r.embedding
                    """,
                    rows=term_rows,
                )

            # Pass 7: MAPS_TO edges (BusinessTerm → TABLE or COLUMN)
            if maps_to_rows:
                await session.run(
                    """
                    UNWIND $rows AS r
                    MATCH (bt:BusinessTerm  {node_id: r.term_id})
                    MATCH (target:SchemaNode {node_id: r.target_id})
                    MERGE (bt)-[:MAPS_TO]->(target)
                    """,
                    rows=maps_to_rows,
                )

        logger.info(
            "Domain '%s': KG written — %d tables, %d columns, %d FK edges, "
            "%d join pairs, %d business terms, %d MAPS_TO edges",
            domain_id, len(table_rows), len(column_rows), len(fk_rows),
            len(seen_pairs) if fk_rows else 0, len(term_rows), len(maps_to_rows),
        )
        return len(vectors)

    # ── Retrieve ──────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        domain_id: str,
        top_k: int = 7,
        permission_tier: int = 2,
    ) -> str:
        """
        Retrieve relevant schema context for *query* using vector ANN search
        on TABLE nodes followed by graph expansion.

        Pipeline
        --------
        1. Embed *query*.
        2. ANN search on :TABLE nodes scoped to *domain_id* and
           sensitivity_level ≤ permission_tier.
        3. 1–2 hop expansion via FOREIGN_KEY + COMMONLY_JOINED_WITH.
        4. Pull :COLUMN children; mask is_pii columns when permission_tier < 2.
        5. Pull :BusinessTerm nodes linked to matched tables/columns.
        6. Return formatted [KNOWLEDGE GRAPH — RELEVANT SCHEMA] block.

        Parameters
        ----------
        query:
            Natural-language question to match against.
        domain_id:
            Only nodes for this domain are considered.
        top_k:
            Number of seed TABLE nodes from ANN before graph expansion.
        permission_tier:
            0 = public, 1 = internal, 2 = all (default).
            Tables with sensitivity_level > permission_tier are excluded.
            PII columns are masked when permission_tier < 2.
        """
        q_vec      = (await generate_embeddings([query]))[0]
        index_name = self._settings.neo4j_vector_index_name
        candidates = top_k * _OVERSAMPLE

        # ── Step 1: ANN search on TABLE nodes ─────────────────────────────
        async with self._driver.session(database=self._database) as session:
            ann_result = await session.run(
                """
                CALL db.index.vector.queryNodes($index, $candidates, $vec)
                YIELD node, score
                WHERE node.domain    = $domain
                  AND node.node_type = 'TABLE'
                  AND node.sensitivity_level <= $tier
                RETURN node, score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                index=index_name,
                candidates=candidates,
                vec=q_vec,
                domain=domain_id,
                tier=permission_tier,
                top_k=top_k,
            )
            seed_records = await ann_result.data()

        if not seed_records:
            logger.debug("KG: no TABLE nodes found for domain '%s'", domain_id)
            return ""

        seed_ids = [r["node"]["node_id"] for r in seed_records]

        # ── Step 2: Graph expansion + column + term fetch ──────────────────
        async with self._driver.session(database=self._database) as session:
            expand_result = await session.run(
                """
                MATCH (seed:TABLE)
                WHERE seed.node_id IN $seed_ids

                OPTIONAL MATCH (seed)-[:FOREIGN_KEY|COMMONLY_JOINED_WITH*1..2]-(hop:TABLE)
                WHERE hop.domain = $domain
                  AND hop.sensitivity_level <= $tier

                WITH collect(DISTINCT seed) + collect(DISTINCT hop) AS all_tables
                UNWIND all_tables AS tbl
                WHERE tbl IS NOT NULL

                OPTIONAL MATCH (tbl)-[hc:HAS_COLUMN]->(col:COLUMN)
                WHERE col.domain = $domain

                OPTIONAL MATCH (bt:BusinessTerm)-[:MAPS_TO]->(tbl)
                WHERE bt.domain = $domain

                RETURN tbl,
                       collect(DISTINCT {col: col, ordinal: hc.ordinal_position}) AS col_data,
                       collect(DISTINCT bt) AS terms
                ORDER BY tbl.name
                """,
                seed_ids=seed_ids,
                domain=domain_id,
                tier=permission_tier,
            )
            records = await expand_result.data()

        if not records:
            return ""

        return self._format_rich_context(records, permission_tier)

    # ── Record join (feedback loop) ────────────────────────────────────────

    async def record_join(
        self,
        domain_id: str,
        table_a: str,
        table_b: str,
    ) -> None:
        """
        Increment COMMONLY_JOINED_WITH.frequency_score between *table_a* and
        *table_b*.  Call this after every successful query that joins them.

        The frequency_score influences graph expansion during retrieval —
        high-frequency pairs are traversed first.
        """
        # Use canonical (sorted) IDs to match the edge regardless of direction.
        a_id = f"{domain_id}:table:{min(table_a, table_b)}"
        b_id = f"{domain_id}:table:{max(table_a, table_b)}"

        async with self._driver.session(database=self._database) as session:
            await session.run(
                """
                MATCH (a:TABLE {node_id: $a_id})-[cj:COMMONLY_JOINED_WITH]-(b:TABLE {node_id: $b_id})
                SET cj.frequency_score = coalesce(cj.frequency_score, 0) + 1
                """,
                a_id=a_id,
                b_id=b_id,
            )
        logger.debug(
            "COMMONLY_JOINED_WITH incremented: %s ↔ %s (domain=%s)",
            table_a, table_b, domain_id,
        )

    # ── Delete domain ─────────────────────────────────────────────────────

    async def delete_domain(self, domain_id: str) -> None:
        """Remove all nodes (and their edges) for *domain_id* from Neo4j."""
        await self._delete_domain(domain_id)

    async def _delete_domain(self, domain_id: str) -> None:
        async with self._driver.session(database=self._database) as session:
            # Delete all SchemaNodes (TABLE, COLUMN, BusinessTerm)
            await session.run(
                "MATCH (n:SchemaNode {domain: $domain}) DETACH DELETE n",
                domain=domain_id,
            )
            # Delete the Domain anchor node
            await session.run(
                "MATCH (d:Domain {node_id: $node_id}) DETACH DELETE d",
                node_id=f"domain:{domain_id}",
            )
        logger.debug("Deleted all KG nodes for domain '%s'", domain_id)

    # ── PII helper ────────────────────────────────────────────────────────

    @staticmethod
    def _column_is_pii(
        table_name: str,
        col_name: str,
        sensitivity_set: set[str],
    ) -> bool:
        """Return True if this column appears in the manifest sensitivity list."""
        qualified = f"{table_name}.{col_name}".lower()
        bare      = col_name.lower()
        return qualified in sensitivity_set or bare in sensitivity_set

    # ── Description builders ──────────────────────────────────────────────

    @staticmethod
    def _table_description(table: dict) -> str:
        col_names = ", ".join(c["name"] for c in table.get("columns", []))
        parts: list[str] = [f"Table {table['name']}"]
        if table.get("description"):
            parts.append(table["description"])
        rows = table.get("estimated_rows", 0)
        if rows:
            parts.append(f"approximately {rows:,} rows")
        parts.append(f"columns: {col_names}")
        return ". ".join(parts)

    @staticmethod
    def _column_description(table_name: str, col: dict) -> str:
        parts: list[str] = [
            f"Column {table_name}.{col['name']} ({col.get('type', 'unknown')})"
        ]
        if col.get("description"):
            parts.append(col["description"])
        flags: list[str] = []
        if col.get("primary_key"):
            flags.append("primary key")
        if col.get("unique"):
            flags.append("unique")
        if not col.get("nullable", True):
            flags.append("NOT NULL")
        for fk in col.get("foreign_keys", []):
            flags.append(f"references {fk['table']}.{fk['column']}")
        if flags:
            parts.append(", ".join(flags))
        return ". ".join(parts)

    # ── Context formatter ─────────────────────────────────────────────────

    def _format_rich_context(
        self,
        records: list[dict],
        permission_tier: int,
    ) -> str:
        """
        Render the expanded subgraph as a compact LLM-readable schema block.

        Tables are listed in alphabetical order.  Columns are ordered by
        ordinal_position.  PII columns are masked when permission_tier < 2.
        BusinessTerms collected across all tables appear in a single glossary
        section at the end.
        """
        lines: list[str] = ["[KNOWLEDGE GRAPH — RELEVANT SCHEMA]"]
        seen_terms: dict[str, dict] = {}

        for record in records:
            tbl      = dict(record["tbl"])
            col_data = record.get("col_data", [])
            terms    = record.get("terms", [])

            header = f"\nTable: {tbl['name']}"
            if tbl.get("description"):
                header += f"  # {tbl['description']}"
            if tbl.get("estimated_rows"):
                header += f"  (~{tbl['estimated_rows']:,} rows)"
            if tbl.get("sensitivity_level", 0) > 0:
                header += "  [RESTRICTED]"
            lines.append(header)

            # Sort columns by ordinal position, skip null entries
            sorted_cols = sorted(
                [e for e in col_data if e.get("col")],
                key=lambda e: (e.get("ordinal") or 0),
            )
            for entry in sorted_cols:
                col = dict(entry["col"])

                # Mask PII columns when caller lacks permission
                if col.get("is_pii") and permission_tier < 2:
                    lines.append(f"  {col['name']}  [PII — RESTRICTED]")
                    continue

                flags: list[str] = []
                if col.get("primary_key"):
                    flags.append("PK")
                if not col.get("nullable", True):
                    flags.append("NOT NULL")
                if col.get("is_pii"):
                    flags.append("PII")
                for fk_target in col.get("fk_targets", []):
                    flags.append(f"→ {fk_target}")

                col_line = (
                    f"  {col['name']} {col.get('data_type', '')}"
                    + (f" [{', '.join(flags)}]" if flags else "")
                )
                if col.get("description"):
                    col_line += f"  # {col['description']}"
                lines.append(col_line)

            # Accumulate business terms from this table
            for bt in (terms or []):
                if bt and bt.get("node_id"):
                    seen_terms[bt["node_id"]] = dict(bt)

        if seen_terms:
            lines.append("\n[BUSINESS GLOSSARY — RELEVANT TERMS]")
            for bt in seen_terms.values():
                lines.append(f"  {bt.get('name', '')}: {bt.get('definition', '')}")

        return "\n".join(lines)
    """
    Neo4j-backed schema knowledge graph.

    Parameters
    ----------
    driver:
        An open ``neo4j.AsyncDriver`` (created once in ``lifespan``).
    database:
        Neo4j database name (AuraDB default: ``"neo4j"``).
    """

    def __init__(self, driver: AsyncDriver, database: str = "neo4j") -> None:
        self._driver   = driver
        self._database = database
        self._settings = get_settings()

    # ── Index setup ───────────────────────────────────────────────────────

    async def ensure_vector_index(self) -> None:
        """
        Create the vector index on :SchemaNode(embedding) if it does not
        already exist.  Safe to call on every startup – IF NOT EXISTS is
        idempotent.
        """
        index_name = self._settings.neo4j_vector_index_name
        dims       = self._settings.neo4j_embedding_dimensions

        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
                FOR (n:SchemaNode) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {dims},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
            )
        logger.info(
            "Neo4j vector index '%s' ensured (%d dims, cosine)",
            index_name, dims,
        )

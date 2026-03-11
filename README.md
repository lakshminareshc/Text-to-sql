# Text-to-SQL API

A production-ready natural-language → SQL service powered by **LangGraph**, **Amazon Bedrock** (Claude Sonnet + Titan Embeddings), **Neo4j AuraDB** knowledge graph, and **FastAPI**.

---

## Architecture overview

```
User query (NL)
      │
      ▼
┌─────────────────────────────────────────────┐
│            FastAPI  /api/v1/query            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          LangGraph SQL Agent                 │
│  load_manifest → generate_sql → validate_sql │
└──────┬──────────────────────┬───────────────┘
       │                      │
       ▼                      ▼
┌─────────────┐    ┌──────────────────────────┐
│ YAML Manifest│    │   Schema RAG Service      │
│  (local/S3) │    │  (lazy KG build per domain│
└─────────────┘    │   on first query)         │
                   └──────────┬───────────────┘
                              │
              ┌───────────────┴─────────────┐
              ▼                             ▼
  ┌─────────────────────┐     ┌────────────────────────┐
  │  PostgreSQL (asyncpg)│     │  Neo4j AuraDB          │
  │  Schema extraction  │     │  Knowledge Graph        │
  │  + LLM enrichment   │     │  Vector index (ANN)     │
  └─────────────────────┘     └────────────────────────┘
              │                             │
              └──────────────┬──────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Amazon Bedrock      │
                  │  Claude Sonnet 4.5   │  ← SQL generation + schema enrichment
                  │  Titan Embed V2      │  ← 1024-dim embeddings (cosine)
                  └─────────────────────┘
```

---

## Tech stack

| Layer | Technology |
|---|---|
| API framework | FastAPI (async) |
| Agent orchestration | LangGraph (StateGraph) |
| LLM — SQL generation | `us.anthropic.claude-sonnet-4-5` via Amazon Bedrock |
| LLM — schema enrichment | `bedrock/anthropic.claude-sonnet-4-5-20250514-v1:0` |
| Embeddings | `bedrock/amazon.titan-embed-text-v2:0` (1024 dims, cosine) |
| Knowledge graph | Neo4j AuraDB — vector index `schema-embeddings` |
| Schema source | PostgreSQL (asyncpg) |
| Manifest store | Local FS or S3 (YAML files) |
| LLM client | LiteLLM (provider-agnostic, retries + backoff) |

---

---

## Project layout

```
Text-to-sql/
├── app/
│   ├── main.py                       # FastAPI app + lifespan (Neo4j driver init)
│   ├── config.py                     # Pydantic-settings (all env vars)
│   ├── agents/
│   │   └── sql_agent.py              # LangGraph: load_manifest → generate_sql → validate_sql
│   ├── manifests/
│   │   ├── manifest_loader.py        # Validates 8 required YAML sections
│   │   └── manifest_store.py         # Local-FS / S3 manifest reader
│   ├── models/
│   │   └── schemas.py                # Pydantic request / response models
│   ├── routes/
│   │   ├── query.py                  # POST /api/v1/query   POST /api/v1/embed
│   │   ├── ingest.py                 # POST /api/v1/ingest/{domain}  (schema → Neo4j)
│   │   └── manifests.py              # GET  /api/v1/manifests[/{domain}]
│   └── services/
│       ├── llm_service.py            # LiteLLM wrapper (Bedrock completions + embeddings)
│       ├── schema_extractor.py       # asyncpg introspection + Bedrock LLM enrichment
│       ├── schema_rag.py             # Lazy per-domain KG build orchestrator
│       └── schema_knowledge_graph.py # Neo4j AuraDB read/write engine
├── manifests/                        # Domain YAML files (local store)
│   ├── chinook.yaml
│   ├── ecommerce.yaml
│   ├── hr.yaml
│   └── sales.yaml
├── scripts/
│   └── ingest_schema.py              # CLI helper to trigger ingestion
├── tests/
│   └── test_manifest_loader.py
├── .env.example                      # Copy → .env and fill in secrets
├── requirements.txt
└── run.py                            # Dev server entry-point
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux
# Edit .env – fill in AWS credentials, Neo4j AuraDB, and Postgres details

# 4. Start the server
python run.py
# OR:  uvicorn app.main:app --reload
```

Interactive docs: http://localhost:8000/docs

---

## Ingestion pipeline

Before querying, ingest a domain's schema into Neo4j:

```
POST /api/v1/ingest/{domain}
```

What happens step-by-step:

1. **Load manifest** — reads `manifests/{domain}.yaml` (local or S3)
2. **Extract schema** — asyncpg introspects PostgreSQL `information_schema` (tables, columns, PKs, FKs, row counts)
3. **Enrich descriptions** — Bedrock Claude Sonnet fills in missing `pg_description` entries for tables and columns
4. **Build Knowledge Graph** — writes to Neo4j AuraDB:
   - `:Domain` anchor node
   - `:SchemaNode:TABLE` nodes — with `sensitivity_level` derived from manifest sensitivity list
   - `:SchemaNode:COLUMN` nodes — with `is_pii`, `ordinal_position`, FK targets
   - `:SchemaNode:BusinessTerm` nodes — from manifest `business_glossary`
   - Relationships: `BELONGS_TO`, `HAS_COLUMN`, `FOREIGN_KEY`, `COMMONLY_JOINED_WITH`, `MAPS_TO`
5. **Embed** — all node descriptions embedded via Bedrock Titan Embed V2 (1024 dims) and stored in Neo4j vector index `schema-embeddings`

---

## Query pipeline

```
POST /api/v1/query
```

What happens step-by-step:

1. **Load manifest** — persona, business rules, sensitivity list, row limits, few-shot examples
2. **Schema RAG** — ANN vector search on `:TABLE` nodes → 1–2 hop graph expansion via `FOREIGN_KEY` / `COMMONLY_JOINED_WITH` → returns `[KNOWLEDGE GRAPH — RELEVANT SCHEMA]` block
3. **Semantic few-shot** — top-K most relevant examples retrieved by cosine similarity from manifest
4. **Generate SQL** — Bedrock Claude Sonnet receives: persona + schema + KG context + glossary + rules + few-shot → returns `{"sql": "...", "explanation": "..."}`
5. **Validate** — sensitivity guard (rejects PII column references) + row-limit enforcement

---

## Environment variables

All variables are documented in [.env.example](.env.example).  
Key ones:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` | ✅ | — | Bedrock + S3 access |
| `AWS_SECRET_ACCESS_KEY` | ✅ | — | Bedrock + S3 access |
| `AWS_REGION` | no | `us-east-1` | Bedrock region |
| `LLM_MODEL` | no | `claude-sonnet-4-5` | SQL generation model |
| `ENRICH_LLM_MODEL` | no | `bedrock/anthropic.claude-sonnet-4-5-20250514-v1:0` | Schema enrichment model |
| `EMBEDDING_MODEL` | no | `bedrock/amazon.titan-embed-text-v2:0` | Embedding model |
| `NEO4J_URI` | ✅ | — | AuraDB connection URI |
| `NEO4J_USER` | ✅ | — | Neo4j username |
| `NEO4J_PASSWORD` | ✅ | — | Neo4j password |
| `NEO4J_DATABASE` | no | `neo4j` | AuraDB database name |
| `NEO4J_EMBEDDING_DIMENSIONS` | no | `1024` | Must match embedding model |
| `POSTGRES_HOST` | ✅ for ingest | — | Schema source DB |
| `POSTGRES_PASSWORD` | ✅ for ingest | — | Schema source DB password |
| `MANIFEST_STORE_TYPE` | no | `local` | `local` or `s3` |
| `CHECKPOINTER_TYPE` | no | `memory` | `memory` or `postgres` |

---

## API endpoints

### `POST /api/v1/ingest/{domain}`
Ingest a domain's PostgreSQL schema into the Neo4j knowledge graph.

```bash
curl -X POST http://localhost:8000/api/v1/ingest/chinook
```

Response:
```json
{
  "domain": "chinook",
  "tables_ingested": 11,
  "embeddings_created": 87,
  "status": "ok"
}
```

### `POST /api/v1/query`
Convert natural language to SQL.

```json
{
  "query": "Top 5 customers by total invoice amount",
  "domain": "chinook",
  "session_id": "optional-uuid",
  "execute": false
}
```

Response:
```json
{
  "sql": "SELECT c.first_name, c.last_name, SUM(i.total) AS total_spent ...",
  "explanation": "Sums invoice totals per customer and returns the top 5.",
  "domain": "chinook",
  "session_id": "uuid"
}
```

### `POST /api/v1/embed`
Generate embeddings via Bedrock Titan Embed V2.

```json
{ "texts": ["some text to embed"] }
```

### `GET /api/v1/manifests`
List all available domain manifests.

### `GET /api/v1/manifests/{domain}`
Return the raw YAML manifest for a domain.

### `GET /api/v1/ingest/{domain}/schema`
Return the cached schema for a domain (last extracted from Postgres).

### `DELETE /api/v1/ingest/{domain}`
Remove all Neo4j nodes for a domain (full reset).

---

## Domain manifest structure

Create `manifests/<domain>.yaml` with all 8 required sections:

```yaml
domain: my_domain
description: "Short description"

persona:
  role: "Expert SQL Analyst"
  tone: "professional and precise"
  expertise: "Deep knowledge of the my_domain database schema."

database_connection:
  type: postgres
  host: "${POSTGRES_HOST}"
  port: 5432
  database: my_db

business_glossary:
  "active customers": "customers WHERE status = 'active'"
  "net revenue": "SUM(amount) - SUM(refunds)"

business_rules:
  - "Always filter soft-deleted records with deleted_at IS NULL"
  - "Use UTC timestamps for all date comparisons"

sensitivity:
  - users.password_hash
  - users.ssn

row_limits:
  default: 100
  max: 1000

few_shot_examples:
  - question: "How many active customers do we have?"
    sql: "SELECT COUNT(*) FROM customers WHERE status = 'active' AND deleted_at IS NULL LIMIT 100;"

tables:                    # optional — used as fallback when Postgres is unreachable
  - name: customers
    description: "Registered customers"
    columns:
      - name: id
        type: INTEGER
        primary_key: true
```

---

## Neo4j knowledge graph model

```
(:Domain)
    └─BELONGS_TO─┐
                 ▼
           (:SchemaNode:TABLE)
           • sensitivity_level (0=public, 1=internal, 2=restricted)
           • embedding (1024-dim Titan V2)
                 │
          HAS_COLUMN
                 ▼
           (:SchemaNode:COLUMN)
           • is_pii (bool)
           • ordinal_position

(:SchemaNode:TABLE)─FOREIGN_KEY {from_col, to_col}─▶(:SchemaNode:TABLE)
(:SchemaNode:TABLE)─COMMONLY_JOINED_WITH {frequency_score}─(:SchemaNode:TABLE)
(:SchemaNode:BusinessTerm)─MAPS_TO─▶(:SchemaNode:TABLE | :COLUMN)
```

`COMMONLY_JOINED_WITH.frequency_score` increments automatically every time a generated SQL query joins two tables.

---

## Read-only PostgreSQL user

```sql
CREATE USER readonly_user WITH PASSWORD 'strong_password_here';
GRANT CONNECT ON DATABASE your_db TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO readonly_user;
```

Then set `POSTGRES_READONLY_*` variables in your `.env`.

---

## Switching to the PostgreSQL checkpointer (production)

In [app/agents/sql_agent.py](app/agents/sql_agent.py), replace the `MemorySaver` block with the commented-out `AsyncPostgresSaver` snippet. Install the extra dependency:

```bash
pip install "langgraph-checkpoint-postgres" psycopg[binary]
```

---

## Bedrock model routing

| Purpose | Model ID | Dims |
|---|---|---|
| SQL generation | `us.anthropic.claude-sonnet-4-5` | — |
| Schema enrichment | `bedrock/anthropic.claude-sonnet-4-5-20250514-v1:0` | — |
| Embeddings | `bedrock/amazon.titan-embed-text-v2:0` | 1024 |

Switch to a different provider without code changes — just update `EMBEDDING_MODEL` and `ENRICH_LLM_MODEL` in `.env`. Supported alternatives (via LiteLLM):

| Provider | Embedding model | Dims |
|---|---|---|
| OpenAI | `openai/text-embedding-3-large` | 3072 |
| Google | `gemini/text-embedding-004` | 768 |

> ⚠️ If you change `EMBEDDING_MODEL`, also update `NEO4J_EMBEDDING_DIMENSIONS` and re-ingest all domains (existing vectors will be incompatible).


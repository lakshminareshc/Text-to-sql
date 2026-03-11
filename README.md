# Text-to-SQL API

Natural-language → SQL service powered by **LangGraph**, **LiteLLM**, and **claude-sonnet-4-5**.

---

## Project layout

```
Text-to-sql/
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── config.py               # Pydantic-settings (all env vars)
│   ├── agents/
│   │   └── sql_agent.py        # LangGraph graph (load_manifest → generate_sql)
│   ├── manifests/
│   │   └── manifest_store.py   # Local-FS / S3 manifest loader
│   ├── models/
│   │   └── schemas.py          # Pydantic request / response models
│   ├── routes/
│   │   ├── query.py            # POST /api/v1/query  POST /api/v1/embed
│   │   └── manifests.py        # GET  /api/v1/manifests[/{domain}]
│   └── services/
│       └── llm_service.py      # LiteLLM wrapper (completion + embeddings)
├── manifests/                  # Domain YAML files (local store)
│   ├── ecommerce.yaml
│   └── hr.yaml
├── .env.example                # Copy → .env and fill in secrets
├── requirements.txt
└── run.py                      # Dev server entry-point
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
# Edit .env – fill in ANTHROPIC_API_KEY (required), others as needed

# 4. Start the server
python run.py
# OR:  uvicorn app.main:app --reload
```

Interactive docs: http://localhost:8000/docs

---

## Environment variables

All variables are documented in [.env.example](.env.example).  
Key ones:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | claude-sonnet-4-5 |
| `OPENAI_API_KEY` | ✅ for `/embed` | — | text-embedding-3-large |
| `MANIFEST_STORE_TYPE` | no | `local` | `local` or `s3` |
| `MANIFEST_LOCAL_PATH` | no | `manifests` | Path to YAML files |
| `CHECKPOINTER_TYPE` | no | `memory` | `memory` or `postgres` |
| `POSTGRES_READONLY_*` | when `execute=true` | — | Read-only query execution |

---

## API endpoints

### `POST /api/v1/query`
Convert natural language to SQL.

```json
{
  "query": "Top 5 customers by revenue last month",
  "domain": "ecommerce",
  "session_id": "optional-uuid",
  "execute": false
}
```

### `POST /api/v1/embed`
Generate embeddings via text-embedding-3-large.

```json
{ "texts": ["some text to embed"] }
```

### `GET /api/v1/manifests`
List all available domain manifests.

### `GET /api/v1/manifests/{domain}`
Return the raw YAML manifest for a domain.

---

## Adding a domain manifest

Create `manifests/<domain>.yaml` following the structure in [manifests/ecommerce.yaml](manifests/ecommerce.yaml).

```yaml
domain: my_domain
description: "Short description"
tables:
  - name: my_table
    description: "What this table stores"
    columns:
      - name: id
        type: INTEGER
        primary_key: true
        description: "PK"
```

---

## Read-only PostgreSQL user

Run this once against your PostgreSQL instance:

```sql
CREATE USER readonly_user WITH PASSWORD 'strong_password_here';
GRANT CONNECT ON DATABASE your_db TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT ON TABLES TO readonly_user;
```

Then set the `POSTGRES_READONLY_*` variables in your `.env`.

---

## Switching to the PostgreSQL checkpointer (production)

In [app/agents/sql_agent.py](app/agents/sql_agent.py), replace the `MemorySaver` block
with the commented-out `AsyncPostgresSaver` snippet.  Install the extra dependency:

```bash
pip install "langgraph-checkpoint-postgres" psycopg[binary]
```

---

## LiteLLM model routing

Default configuration in [app/services/llm_service.py](app/services/llm_service.py):

| Purpose | Model | Provider |
|---|---|---|
| SQL generation | `claude-sonnet-4-5` | Anthropic |
| Embeddings | `text-embedding-3-large` | OpenAI |

Retries: **3 attempts**, exponential back-off 2 s → 30 s (configurable via env vars).

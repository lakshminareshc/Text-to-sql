from typing import Any
from pydantic import BaseModel, Field


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question to convert to SQL")
    domain: str = Field(..., description="Domain / manifest name that defines the schema")
    session_id: str | None = Field(
        default=None,
        description="Conversation session ID for multi-turn history; auto-generated if omitted",
    )
    execute: bool = Field(
        default=False,
        description="If True, execute the generated SQL against the read-only database",
    )


class QueryResponse(BaseModel):
    sql: str = Field(..., description="Generated SQL query")
    explanation: str = Field(..., description="Plain-English explanation of what the SQL does")
    domain: str
    session_id: str
    rows: list[dict[str, Any]] | None = Field(
        default=None,
        description="Result rows when execute=True",
    )


# ── Embeddings ────────────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    texts: str | list[str] = Field(..., description="One or more texts to embed")
    model: str | None = Field(default=None, description="Override the default embedding model")


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    usage: dict[str, int] | None = None


# ── Manifests ─────────────────────────────────────────────────────────────────

class ManifestSummary(BaseModel):
    domain: str
    description: str | None = None
    table_count: int


class ManifestDetail(BaseModel):
    domain: str
    manifest: dict[str, Any]


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    domain: str
    status: str = Field(..., description="'ingested' | 'invalidated'")
    table_count: int
    column_count: int
    fk_count: int
    embeddings_generated: int = 0
    duration_ms: int = 0
    message: str


class SchemaPreviewResponse(BaseModel):
    domain: str
    table_count: int
    column_count: int
    tables: list[dict[str, Any]]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    checkpointer: str
    manifest_store: str

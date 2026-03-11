from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Text-to-SQL API"
    app_version: str = "0.1.0"
    debug: bool = False

    # ── Anthropic / LLM ──────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-5"          # used for SQL generation
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048
    llm_max_retries: int = 3                       # tenacity + LiteLLM retries
    llm_retry_min_wait: float = 2.0               # seconds (exponential backoff)
    llm_retry_max_wait: float = 30.0

    # ── Schema description enrichment LLM ────────────────────────────────────
    # Switch provider purely via .env — no code changes needed:
    #   Bedrock Claude:  ENRICH_LLM_MODEL=bedrock/anthropic.claude-sonnet-4-5-20250514-v1:0
    #   Groq (fallback): ENRICH_LLM_MODEL=groq/llama-3.3-70b-versatile
    #   Direct Anthropic:ENRICH_LLM_MODEL=anthropic/claude-sonnet-4-5
    groq_api_key: str = ""
    enrich_llm_model: str = "bedrock/anthropic.claude-sonnet-4-5-20250514-v1:0"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Switch provider purely via .env — no code changes needed:
    #   Bedrock Titan V2: EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0   NEO4J_EMBEDDING_DIMENSIONS=1024
    #   Gemini (free):    EMBEDDING_MODEL=gemini/text-embedding-004               NEO4J_EMBEDDING_DIMENSIONS=768
    #   OpenAI:           EMBEDDING_MODEL=openai/text-embedding-3-large           NEO4J_EMBEDDING_DIMENSIONS=3072
    gemini_api_key: str = ""
    openai_api_key: str = ""
    embedding_model: str = "bedrock/amazon.titan-embed-text-v2:0"

    # ── Manifest store ────────────────────────────────────────────────────────
    manifest_store_type: Literal["local", "s3"] = "local"
    manifest_local_path: str = "manifests"        # relative to project root

    # S3 – only required when manifest_store_type = "s3"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = ""
    s3_manifest_prefix: str = "manifests/"        # key prefix inside the bucket

    # ── PostgreSQL (read-write – app writes, not exposed to LLM) ────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = ""
    postgres_user: str = ""
    postgres_password: str = ""

    # PostgreSQL read-only user (used by LLM-generated SQL execution)
    postgres_readonly_host: str = "localhost"
    postgres_readonly_port: int = 5432
    postgres_readonly_db: str = ""
    postgres_readonly_user: str = ""
    postgres_readonly_password: str = ""

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # ── Neo4j AuraDB ──────────────────────────────────────────────────────────
    # URI format for AuraDB:  neo4j+s://<instance-id>.databases.neo4j.io
    # URI format for local Docker: bolt://localhost:7687
    neo4j_uri: str = "neo4j+s://xxxxxxxx.databases.neo4j.io"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"                    # AuraDB default database name
    neo4j_vector_index_name: str = "schema-embeddings"
    neo4j_embedding_dimensions: int = 1024             # amazon.titan-embed-text-v2:0 → 1024

    # ── LangGraph checkpointer ────────────────────────────────────────────────
    # Set to "memory" for development; switch to "postgres" in production
    checkpointer_type: Literal["memory", "postgres"] = "memory"

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_readonly_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_readonly_user}:{self.postgres_readonly_password}"
            f"@{self.postgres_readonly_host}:{self.postgres_readonly_port}/{self.postgres_readonly_db}"
        )

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    return Settings()

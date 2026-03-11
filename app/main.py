"""
FastAPI application entry-point
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from neo4j import AsyncGraphDatabase

from app.config import get_settings
from app.models.schemas import HealthResponse
from app.routes import ingest, manifests, query
from app.services.schema_knowledge_graph import SchemaKnowledgeGraph
from app.services.schema_rag import init_driver

logging.basicConfig(
    level=logging.DEBUG if get_settings().debug else logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("LLM model       : %s", settings.llm_model)
    logger.info("Embedding model : %s", settings.embedding_model)
    logger.info("Manifest store  : %s", settings.manifest_store_type)
    logger.info("Checkpointer    : %s", settings.checkpointer_type)

    # ── Neo4j AuraDB driver ───────────────────────────────────────────────────
    # Non-fatal: if AuraDB is paused or unreachable, the server still starts.
    # Ingest and query endpoints will return 500 until Neo4j is reachable.
    neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
        max_connection_pool_size=50,
    )
    try:
        await neo4j_driver.verify_connectivity()
        logger.info("Neo4j connected  : %s (db=%s)", settings.neo4j_uri, settings.neo4j_database)

        # Ensure vector index exists (idempotent — IF NOT EXISTS)
        kg_bootstrap = SchemaKnowledgeGraph(neo4j_driver, settings.neo4j_database)
        await kg_bootstrap.ensure_vector_index()

        # Register driver with the Schema RAG service
        init_driver(neo4j_driver, settings.neo4j_database)
    except Exception as exc:
        logger.warning(
            "Neo4j unavailable at startup (%s). "
            "Server will start without Neo4j — resume your AuraDB instance at "
            "https://console.neo4j.io and restart the server.",
            exc,
        )

    yield

    await neo4j_driver.close()
    logger.info("Neo4j driver closed.")
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Text-to-SQL API powered by LangGraph agents and LiteLLM "
        "(claude-sonnet-4-5 + text-embedding-3-large)."
    ),
    lifespan=lifespan,
)

# ── CORS (tighten origins in production) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(query.router,     prefix="/api/v1", tags=["query"])
app.include_router(manifests.router, prefix="/api/v1", tags=["manifests"])
app.include_router(ingest.router,    prefix="/api/v1", tags=["ingest"])


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        checkpointer=settings.checkpointer_type,
        manifest_store=settings.manifest_store_type,
    )

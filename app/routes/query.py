"""
Routes: /api/v1/query  and  /api/v1/embed
"""
import logging

from fastapi import APIRouter, HTTPException

from app.agents.sql_agent import run_sql_agent
from app.models.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.llm_service import generate_embeddings
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


@router.post("/query", response_model=QueryResponse, summary="Natural language → SQL")
async def natural_language_to_sql(request: QueryRequest) -> QueryResponse:
    """
    Convert a natural-language question to a SQL query using the LangGraph agent.

    - **query**: the question in plain English
    - **domain**: name of the domain manifest (e.g. `ecommerce`)
    - **session_id**: optional – supply the same ID across turns for conversation history
    - **execute**: set to `true` to run the SQL against the read-only database
    """
    try:
        result = await run_sql_agent(
            query=request.query,
            domain=request.domain,
            session_id=request.session_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("SQL agent error: %s", exc)
        raise HTTPException(status_code=500, detail="SQL generation failed.") from exc

    if result.get("validation_error"):
        raise HTTPException(status_code=422, detail=result["validation_error"])

    rows = None
    if request.execute:
        # Placeholder – wire up the async read-only DB connection here.
        # Example (using asyncpg):
        #   import asyncpg
        #   conn = await asyncpg.connect(settings.postgres_readonly_dsn)
        #   rows = [dict(r) for r in await conn.fetch(result["sql"])]
        raise HTTPException(
            status_code=501,
            detail="SQL execution is not yet configured. Set up the read-only DB connection.",
        )

    return QueryResponse(
        sql=result["sql"],
        explanation=result["explanation"],
        domain=request.domain,
        session_id=result["session_id"],
        rows=rows,
    )


@router.post("/embed", response_model=EmbeddingResponse, summary="Generate text embeddings")
async def embed_texts(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Embed one or more texts using **text-embedding-3-large** via LiteLLM → OpenAI.
    """
    texts = [request.texts] if isinstance(request.texts, str) else request.texts
    try:
        vectors = await generate_embeddings(texts, model=request.model or None)
    except Exception as exc:
        logger.exception("Embedding error: %s", exc)
        raise HTTPException(status_code=500, detail="Embedding generation failed.") from exc

    return EmbeddingResponse(
        embeddings=vectors,
        model=request.model or settings.embedding_model,
    )

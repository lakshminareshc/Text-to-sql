"""
LiteLLM service
───────────────
• claude-sonnet-4-5 (Bedrock)            → text generation / SQL synthesis
• amazon.titan-embed-text-v2:0 (Bedrock) → vector embeddings (1024 dims)
• claude-sonnet-4-5 (Bedrock)            → schema description enrichment

All Bedrock calls authenticate via AWS SigV4 using the AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION env vars (injected from Settings).

Both completion and embedding functions are wrapped with Tenacity for
exponential-backoff retries (max 3 attempts, 2 s→30 s window).
LiteLLM's own internal retry counter is set to the same value so the two
layers don't fight each other.
"""
import logging
import os

import litellm
from litellm import acompletion, aembedding
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ── Inject API keys / AWS credentials into os.environ ───────────────────────
# pydantic-settings reads .env into Python objects but does NOT set os.environ.
# LiteLLM looks up credentials via os.environ, so we set them explicitly here.
if settings.gemini_api_key:
    os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
if settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
if settings.groq_api_key:
    os.environ["GROQ_API_KEY"] = settings.groq_api_key
if settings.anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
# Bedrock uses AWS SigV4 auth — LiteLLM picks these up automatically.
if settings.aws_access_key_id:
    os.environ["AWS_ACCESS_KEY_ID"]     = settings.aws_access_key_id
if settings.aws_secret_access_key:
    os.environ["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
if settings.aws_region:
    os.environ["AWS_DEFAULT_REGION"]    = settings.aws_region

# ── LiteLLM global config ─────────────────────────────────────────────────────
litellm.set_verbose = settings.debug
litellm.num_retries = settings.llm_max_retries   # LiteLLM-level retries (transport)
litellm.drop_params = True                        # ignore unknown params silently

# Fully-qualified model names understood by LiteLLM router
LLM_MODEL = f"anthropic/{settings.llm_model}"
EMBEDDING_MODEL = settings.embedding_model   # already prefixed: e.g. gemini/text-embedding-004

# ── API key resolver ──────────────────────────────────────────────────────────
_API_KEYS: dict[str, str] = {
    "anthropic": settings.anthropic_api_key,
    "groq":      settings.groq_api_key,
    "openai":    settings.openai_api_key,
    "gemini":    settings.gemini_api_key,
}

def _api_key_for(model: str) -> str | None:
    """Return the correct API key for *model* based on its provider prefix.

    Works for any LiteLLM model string: ``groq/llama-3.3-70b-versatile``,
    ``anthropic/claude-sonnet-4-5``, ``openai/gpt-4o``, etc.
    Returns ``None`` (LiteLLM reads from env) when the key is empty or the
    provider is not in the registry.
    """
    provider = model.split("/")[0] if "/" in model else ""
    key = _API_KEYS.get(provider, "")
    return key or None

# ── Retry decorator (shared between completion & embedding) ───────────────────
_RETRYABLE = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
)

def _llm_retry():
    """Factory so each function gets its own Tenacity state."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(
            multiplier=1,
            min=settings.llm_retry_min_wait,
            max=settings.llm_retry_max_wait,
        ),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=lambda rs: logger.warning(
            "LLM call failed, retrying (attempt %d): %s",
            rs.attempt_number,
            rs.outcome.exception(),
        ),
    )


# ── Completion ────────────────────────────────────────────────────────────────

async def generate_completion(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Send a chat-completion request.

    - ``bedrock/...`` models → boto3 directly (preserves us. inference profile
      prefix that LiteLLM otherwise strips).
    - All other models → LiteLLM (Anthropic, Groq, OpenAI, etc.).

    Returns the assistant message content as a plain string.
    """
    resolved_model = model or LLM_MODEL
    if resolved_model.startswith("bedrock/"):
        return await _bedrock_complete(
            messages,
            model=resolved_model,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )
    return await _litellm_complete(
        messages,
        model=resolved_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
    )


async def _bedrock_complete(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Call Bedrock InvokeModel with boto3 directly so the inference profile ID
    (e.g. ``us.anthropic.claude-...``) is passed through unchanged.
    LiteLLM strips the ``us.`` prefix before the API call, causing Bedrock to
    reject on-demand invocations of cross-region inference profiles.
    """
    import asyncio
    import json

    import boto3  # type: ignore[import]

    model_id = model.removeprefix("bedrock/")
    client = boto3.client(
        "bedrock-runtime",
        region_name=settings.aws_region or "us-east-1",
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }

    def _invoke() -> str:
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    logger.info("Bedrock direct invoke: %s", model_id)
    return await asyncio.get_event_loop().run_in_executor(None, _invoke)


@_llm_retry()
async def _litellm_complete(
    messages: list[dict],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Send a chat-completion request via LiteLLM."""
    response = await acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=_api_key_for(model),
    )
    content: str = response.choices[0].message.content  # type: ignore[index]
    return content


# ── Embeddings ────────────────────────────────────────────────────────────────

async def generate_embeddings(
    texts: list[str],
    *,
    model: str | None = None,
) -> list[list[float]]:
    """
    Embed texts using the configured embedding model, routing by prefix:

    - ``bedrock/...``  → LiteLLM + AWS SigV4 (Bedrock Titan / Cohere on Bedrock)
    - ``gemini/...``   → Google AI Studio SDK directly (avoids Vertex AI routing)
    - everything else  → LiteLLM generic (OpenAI, Cohere, etc.)

    Returns a list of float vectors in the same order as ``texts``.
    """
    resolved = model or EMBEDDING_MODEL
    if resolved.startswith("bedrock/"):
        return await _litellm_embed(texts, model=resolved)
    if settings.gemini_api_key and (
        resolved.startswith("gemini/") or "gemini" in resolved.lower()
    ):
        return await _gemini_embed(texts, model=resolved)
    return await _litellm_embed(texts, model=resolved)


async def _gemini_embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Call Google AI Studio embed_content using the new google-genai SDK."""
    import asyncio
    from google import genai  # type: ignore[import]

    client = genai.Client(api_key=settings.gemini_api_key)
    # Strip any prefix: "gemini/text-embedding-004" or "text-embedding-004" both work
    model_name = (model or EMBEDDING_MODEL).removeprefix("gemini/")

    def _embed_sync() -> list[list[float]]:
        result = client.models.embed_content(model=model_name, contents=texts)
        return [e.values for e in result.embeddings]

    return await asyncio.get_event_loop().run_in_executor(None, _embed_sync)


@_llm_retry()
async def _litellm_embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Fallback: embed via LiteLLM (OpenAI, Cohere, etc.)"""
    resolved = model or EMBEDDING_MODEL
    response = await aembedding(
        model=resolved,
        input=texts,
        api_key=_api_key_for(resolved),
    )
    return [item["embedding"] for item in response.data]


# ── Model routing reference (document for router configuration) ───────────────
#
# To use LiteLLM's Router (load-balancing / fallbacks), create a router like:
#
#   from litellm import Router
#   router = Router(
#       model_list=[
#           {
#               "model_name": "default-llm",
#               "litellm_params": {
#                   "model": "anthropic/claude-sonnet-4-5",
#                   "api_key": settings.anthropic_api_key,
#                   "num_retries": 3,
#               },
#           },
#           {
#               "model_name": "default-embedding",
#               "litellm_params": {
#                   "model": "openai/text-embedding-3-large",
#                   "api_key": settings.openai_api_key,
#               },
#           },
#       ],
#       allowed_fails=2,
#       cooldown_time=10,
#   )

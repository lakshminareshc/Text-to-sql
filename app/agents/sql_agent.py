"""
LangGraph SQL-generation agent
──────────────────────────────
Graph nodes
  1. load_manifest  – load + validate domain YAML via ManifestLoader
  2. generate_sql   – build rich prompt from manifest blocks → LLM → structured SQL
  3. validate_sql   – sensitivity guard + row-limit enforcement

Manifest blocks consumed
  • persona              → system-prompt preamble (role / tone / expertise)
  • business_glossary    → SQL fragment substitutions injected into prompt
  • business_rules       → hard constraints injected verbatim into prompt
  • sensitivity          → restricted columns; any reference causes rejection
  • row_limits           → domain-specific LIMIT applied and enforced
  • few_shot_examples    → semantically retrieved Q→SQL pairs injected as examples

Checkpointer
  • Development : MemorySaver (in-process, non-persistent)
  • Production  : swap in AsyncPostgresSaver (see comment below)

The compiled graph is cached as a module-level singleton via get_sql_agent().
"""
from __future__ import annotations

import json
import logging
import operator
import re
import uuid
from typing import Annotated, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from app.manifests.manifest_loader import get_manifest_loader
from app.services.llm_service import generate_completion, generate_embeddings
from app.services.schema_rag import get_schema_context

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    domain: str
    session_id: str
    manifest: dict[str, Any]
    sql: str
    explanation: str
    validation_error: str | None   # set by validate_sql node if query is rejected
    messages: Annotated[list[dict], operator.add]


_USER_TEMPLATE = "{query}"


# ── Schema formatter ─────────────────────────────────────────────────────────

def _format_schema(manifest: dict) -> str:
    lines: list[str] = []
    for table in manifest.get("tables", []):
        tname = table.get("name", "unknown")
        tdesc = table.get("description", "")
        lines.append(f"Table: {tname}  # {tdesc}")
        for col in table.get("columns", []):
            cname = col.get("name", "")
            ctype = col.get("type", "")
            cdesc = col.get("description", "")
            pk    = " [PK]" if col.get("primary_key") else ""
            fk    = f" → {col['foreign_key']}" if col.get("foreign_key") else ""
            lines.append(f"  {cname} {ctype}{pk}{fk}  # {cdesc}")
        lines.append("")
    return "\n".join(lines)


# ── Few-shot semantic retrieval ───────────────────────────────────────────────
#
# On first access for a domain, all example questions are embedded and cached.
# Subsequent queries embed only the user question and pick the top-K by cosine
# similarity.  Falls back to returning the first *top_k* examples if the
# embedding service is unavailable.

_FEW_SHOT_CACHE: dict[str, list[tuple[list[float], dict]]] = {}
_FEW_SHOT_TOP_K = 5


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


async def _retrieve_few_shot_examples(
    query: str,
    manifest: dict,
    domain: str,
    top_k: int = _FEW_SHOT_TOP_K,
) -> list[dict]:
    """
    Return the *top_k* most semantically relevant few-shot examples for *query*.

    Uses cached question embeddings to avoid redundant API calls.  If the
    embedding service is unavailable, the first *top_k* examples are returned.
    """
    examples: list[dict] = manifest.get("few_shot_examples", [])
    if not examples:
        return []
    if len(examples) <= top_k:
        return examples

    try:
        if domain not in _FEW_SHOT_CACHE:
            questions = [ex["question"] for ex in examples]
            embeddings = await generate_embeddings(questions)
            _FEW_SHOT_CACHE[domain] = list(zip(embeddings, examples))

        query_vec = (await generate_embeddings([query]))[0]
        scored = [
            (_cosine_similarity(query_vec, emb), ex)
            for emb, ex in _FEW_SHOT_CACHE[domain]
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [ex for _, ex in scored[:top_k]]

    except Exception as exc:
        logger.warning(
            "Semantic few-shot retrieval failed (%s); using first %d examples.",
            exc, top_k,
        )
        return examples[:top_k]


# ── System prompt builder ─────────────────────────────────────────────────────

async def _build_system_prompt(
    manifest: dict,
    domain: str,
    query: str,
    kg_context: str = "",
) -> str:
    """
    Compose the full system prompt from all manifest blocks:
      persona · schema · KG context · business_glossary · business_rules · few_shot_examples
    """
    # Persona preamble
    persona = manifest.get("persona", {})
    role      = persona.get("role",      "Expert SQL Analyst")
    tone      = persona.get("tone",      "professional and precise")
    expertise = persona.get("expertise", "").strip()

    preamble = (
        f"You are a {role}.\n"
        f"Tone: {tone}.\n"
        f"{expertise}\n"
    )

    # Schema
    schema_text = _format_schema(manifest)

    # Row limits
    row_limits   = manifest.get("row_limits", {})
    default_limit = int(row_limits.get("default", 100))
    max_limit     = int(row_limits.get("max",     1000))

    # Business glossary
    glossary: dict = manifest.get("business_glossary", {})
    if glossary:
        glossary_lines = "\n".join(
            f'  "{term}": {expr}' for term, expr in glossary.items()
        )
        glossary_section = f"BUSINESS GLOSSARY\n{glossary_lines}\n"
    else:
        glossary_section = ""

    # Business rules
    rules: list = manifest.get("business_rules", [])
    if rules:
        rules_lines = "\n".join(f"  - {rule}" for rule in rules)
        rules_section = f"BUSINESS RULES (apply every rule to every query)\n{rules_lines}\n"
    else:
        rules_section = ""

    # Few-shot examples (semantically retrieved)
    examples = await _retrieve_few_shot_examples(query, manifest, domain)
    if examples:
        ex_blocks = "\n\n".join(
            f"Q: {ex['question']}\nSQL:\n{ex['sql'].strip()}"
            for ex in examples
        )
        examples_section = f"FEW-SHOT EXAMPLES\n{ex_blocks}\n"
    else:
        examples_section = ""

    kg_section = f"{kg_context}\n" if kg_context else ""

    return f"""{preamble}
You receive a natural-language question and the database schema for the '{domain}' domain.

SCHEMA:
{schema_text}
{kg_section}
{glossary_section}
{rules_section}
{examples_section}
INSTRUCTIONS:
- Output ONLY valid, read-only SELECT SQL. Never generate INSERT / UPDATE /
  DELETE / DROP / TRUNCATE or any DDL.
- Qualify column names with the table alias when joins are present.
- Use LIMIT {default_limit} unless the user explicitly requests more rows (hard cap: {max_limit}).
- Apply all business rules and glossary definitions above to every query.
- Return your answer as a JSON object with exactly two keys:
    {{"sql": "<query>", "explanation": "<one-sentence plain-English summary>"}}
- Do NOT wrap the JSON in markdown code fences.
"""


# ── SQL sensitivity validator ─────────────────────────────────────────────────

def _check_sensitivity(sql: str, sensitivity: list[str]) -> str | None:
    """
    Return an error message if *sql* references any restricted column name,
    otherwise return None.

    Matching is word-boundary aware to avoid false positives on sub-strings
    (e.g. 'email_verified' should not match the restricted name 'email').
    Bare column names such as 'email' or qualified ones like 'leads.email'
    are both detected.
    """
    sql_lower = sql.lower()
    violations: list[str] = []
    for col in sensitivity:
        # Match the last segment after an optional table prefix
        bare = col.split(".")[-1].lower()
        pattern = rf"(?<![\w.])(?:[\w]+\.)?{re.escape(bare)}(?![\w])"
        if re.search(pattern, sql_lower):
            violations.append(col)
    if violations:
        return (
            f"Query rejected: references restricted / PII column(s): "
            f"{violations}. Reformulate without accessing these fields."
        )
    return None


# ── Row-limit enforcement ─────────────────────────────────────────────────────

def _enforce_row_limit(sql: str, default_limit: int, max_limit: int) -> str:
    """
    Ensure the SQL has a LIMIT clause that does not exceed *max_limit*.

    - If no LIMIT is present, append ``LIMIT {default_limit}``.
    - If a LIMIT is present and exceeds *max_limit*, cap it.
    """
    limit_pattern = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
    match = limit_pattern.search(sql)
    if match:
        current = int(match.group(1))
        if current > max_limit:
            sql = limit_pattern.sub(f"LIMIT {max_limit}", sql)
    else:
        # Strip trailing semicolon, add limit, re-add semicolon
        stripped = sql.rstrip().rstrip(";")
        sql = f"{stripped}\nLIMIT {default_limit};"
    return sql


# ── Parse LLM output ─────────────────────────────────────────────────────────

def _parse_response(text: str) -> tuple[str, str]:
    """Extract (sql, explanation) from the LLM JSON response."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data = json.loads(cleaned)
        return data["sql"].strip(), data["explanation"].strip()
    except (json.JSONDecodeError, KeyError):
        logger.warning("Could not parse LLM JSON response; returning raw output.")
        return cleaned, ""


# ── Graph nodes ───────────────────────────────────────────────────────────────

async def _load_manifest(state: AgentState) -> dict:
    loader = get_manifest_loader()
    manifest = await loader.load(state["domain"])
    logger.debug("Manifest loaded and validated for domain '%s'", state["domain"])
    return {"manifest": manifest}


async def _generate_sql(state: AgentState) -> dict:
    kg_context = await get_schema_context(
        state["domain"], state["manifest"], state["query"]
    )
    system_msg = await _build_system_prompt(
        state["manifest"], state["domain"], state["query"], kg_context
    )
    user_msg = _USER_TEMPLATE.format(query=state["query"])

    messages = [
        {"role": "system", "content": system_msg},
        *state.get("messages", []),
        {"role": "user",   "content": user_msg},
    ]

    raw = await generate_completion(messages)
    sql, explanation = _parse_response(raw)

    return {
        "sql": sql,
        "explanation": explanation,
        "validation_error": None,
        "messages": [
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": raw},
        ],
    }


async def _validate_sql(state: AgentState) -> dict:
    """
    Post-generation validation node:
      1. Sensitivity check  – reject if restricted columns are referenced.
      2. Row-limit guard    – cap or inject LIMIT from the domain's row_limits block.
    """
    manifest = state["manifest"]
    sql      = state["sql"]

    # 1. Sensitivity check
    sensitivity: list[str] = manifest.get("sensitivity", [])
    error = _check_sensitivity(sql, sensitivity)
    if error:
        logger.warning("Sensitivity violation for domain '%s': %s", state["domain"], error)
        return {"validation_error": error, "sql": ""}

    # 2. Row-limit enforcement
    row_limits    = manifest.get("row_limits", {})
    default_limit = int(row_limits.get("default", 100))
    max_limit     = int(row_limits.get("max",     1000))
    safe_sql = _enforce_row_limit(sql, default_limit, max_limit)

    return {"sql": safe_sql, "validation_error": None}


# ── Graph definition ──────────────────────────────────────────────────────────

def _build_graph():
    workflow: StateGraph = StateGraph(AgentState)

    workflow.add_node("load_manifest", _load_manifest)
    workflow.add_node("generate_sql",  _generate_sql)
    workflow.add_node("validate_sql",  _validate_sql)

    workflow.set_entry_point("load_manifest")
    workflow.add_edge("load_manifest", "generate_sql")
    workflow.add_edge("generate_sql",  "validate_sql")
    workflow.add_edge("validate_sql",  END)

    # ── Checkpointer selection ────────────────────────────────────────────────
    # Development default: in-memory (no persistence across restarts).
    #
    # To switch to PostgreSQL in production, replace the block below with:
    #
    #   from psycopg import AsyncConnection
    #   from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    #   from app.config import get_settings
    #   settings = get_settings()
    #   conn = await AsyncConnection.connect(settings.postgres_dsn)
    #   checkpointer = AsyncPostgresSaver(conn)
    #   await checkpointer.setup()
    #
    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent = None


def get_sql_agent():
    global _agent
    if _agent is None:
        _agent = _build_graph()
        logger.info("LangGraph SQL agent compiled (checkpointer=memory).")
    return _agent


# ── Public helper ─────────────────────────────────────────────────────────────

async def run_sql_agent(
    query: str,
    domain: str,
    session_id: str | None = None,
) -> dict[str, str]:
    """
    Run the agent for one turn.

    Returns
    -------
    dict with keys: sql, explanation, session_id
    """
    sid = session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": sid}}

    initial_state: AgentState = {
        "query":            query,
        "domain":           domain,
        "session_id":       sid,
        "manifest":         {},
        "sql":              "",
        "explanation":      "",
        "validation_error": None,
        "messages":         [],
    }

    result = await get_sql_agent().ainvoke(initial_state, config=config)
    return {
        "sql":              result["sql"],
        "explanation":      result["explanation"],
        "session_id":       sid,
        "validation_error": result.get("validation_error"),
    }

"""
ManifestLoader
──────────────
Loads and validates domain YAML manifests from the ManifestStore.

Required top-level sections
---------------------------
  persona              – role, tone, expertise  →  injected as system-prompt preamble
  database_connection  – env-var *names* only; credentials resolved at runtime
  business_glossary    – plain-English term → SQL fragment mapping
  business_rules       – hard SQL constraints injected verbatim into every prompt
  sensitivity          – PII / restricted column names; SQL Validator rejects references
  row_limits           – max rows per domain (keys: default, max)
  few_shot_examples    – 8–12 question→SQL pairs; used for semantic retrieval at query time
  tables               – schema definition (tables + columns)

All required sections must be present and have the correct type, or a
``ManifestValidationError`` is raised with a structured description of the problem.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.manifests.manifest_store import ManifestNotFoundError, ManifestStore  # noqa: F401

logger = logging.getLogger(__name__)

# ── Schema contract ───────────────────────────────────────────────────────────

REQUIRED_SECTIONS: list[str] = [
    "persona",
    "database_connection",
    "business_glossary",
    "business_rules",
    "sensitivity",
    "row_limits",
    "few_shot_examples",
]

_SECTION_TYPES: dict[str, type] = {
    "persona":             dict,
    "database_connection": dict,
    "business_glossary":   dict,
    "business_rules":      list,
    "sensitivity":         list,
    "row_limits":          dict,
    "few_shot_examples":   list
}


# ── Exceptions ────────────────────────────────────────────────────────────────

class ManifestValidationError(Exception):
    """
    Raised when a manifest fails structural validation.

    Attributes
    ----------
    domain  : str        – the domain_id that was requested
    missing : list[str]  – section keys absent from the manifest (may be empty
                           when the error is a type mismatch rather than absence)
    """

    def __init__(
        self,
        domain: str,
        message: str,
        missing: list[str] | None = None,
    ) -> None:
        self.domain = domain
        self.missing: list[str] = missing or []
        super().__init__(f"Manifest '{domain}' validation failed: {message}")


# ── ManifestLoader ────────────────────────────────────────────────────────────

class ManifestLoader:
    """Load and validate domain manifests from a ManifestStore backend."""

    def __init__(self, store: ManifestStore) -> None:
        self._store = store

    async def load(self, domain_id: str) -> dict[str, Any]:
        """
        Fetch and validate the manifest for *domain_id*.

        Raises
        ------
        ManifestNotFoundError
            If no YAML file exists for the domain in the store.
        ManifestValidationError
            If any required section is absent or has the wrong Python type.
        """
        manifest = await self._store.get_manifest(domain_id)
        self._validate(manifest, domain_id)
        logger.info("Manifest loaded and validated for domain '%s'", domain_id)
        return manifest

    # ── Internal validation ───────────────────────────────────────────────────

    def _validate(self, manifest: dict, domain_id: str) -> None:
        """Check presence and type of every required section."""
        missing = [s for s in REQUIRED_SECTIONS if s not in manifest]
        if missing:
            raise ManifestValidationError(
                domain=domain_id,
                message=f"missing required sections: {missing}",
                missing=missing,
            )

        for section, expected_type in _SECTION_TYPES.items():
            value = manifest[section]
            if not isinstance(value, expected_type):
                raise ManifestValidationError(
                    domain=domain_id,
                    message=(
                        f"section '{section}' must be a {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    ),
                )


# ── Database connection resolver ──────────────────────────────────────────────

def resolve_db_connection(connection_block: dict[str, Any]) -> dict[str, str]:
    """
    Resolve a ``database_connection`` block (which stores env-var *names*) into
    actual connection values read from the live process environment.

    Manifests never embed credentials directly.  Instead they store the *names*
    of the environment variables that hold the credentials::

        database_connection:
          type: postgresql
          env_vars:
            host:     SALES_DB_HOST
            port:     SALES_DB_PORT
            database: SALES_DB_NAME
            user:     SALES_DB_USER
            password: SALES_DB_PASSWORD

    Returns a plain ``dict`` with the resolved values plus a ``"type"`` key.

    Raises
    ------
    EnvironmentError
        If any referenced environment variable is absent from the process environment.
    """
    env_vars: dict[str, str] = connection_block.get("env_vars", {})
    resolved: dict[str, str] = {}
    missing_vars: list[str] = []

    for key, var_name in env_vars.items():
        value = os.environ.get(var_name)
        if value is None:
            missing_vars.append(var_name)
        else:
            resolved[key] = value

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables for DB connection: {missing_vars}"
        )

    resolved["type"] = connection_block.get("type", "postgresql")
    return resolved


# ── Singleton factory ─────────────────────────────────────────────────────────

_loader: ManifestLoader | None = None


def get_manifest_loader() -> ManifestLoader:
    """Return the application-level ManifestLoader singleton."""
    global _loader
    if _loader is None:
        from app.manifests.manifest_store import get_manifest_store
        _loader = ManifestLoader(get_manifest_store())
    return _loader

"""
Unit tests: ManifestLoader + resolve_db_connection
───────────────────────────────────────────────────

Run:   pytest tests/test_manifest_loader.py -v
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.manifests.manifest_loader import (
    REQUIRED_SECTIONS,
    ManifestLoader,
    ManifestValidationError,
    resolve_db_connection,
)
from app.manifests.manifest_store import ManifestNotFoundError, ManifestStore


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def valid_manifest() -> dict:
    """A minimal but fully-conformant manifest with all required sections."""
    return {
        "domain": "test",
        "description": "Test domain",
        "persona": {
            "role": "Test Analyst",
            "tone": "neutral",
            "expertise": "General SQL testing.",
        },
        "database_connection": {
            "type": "postgresql",
            "env_vars": {
                "host":     "TEST_DB_HOST",
                "port":     "TEST_DB_PORT",
                "database": "TEST_DB_NAME",
                "user":     "TEST_DB_USER",
                "password": "TEST_DB_PASSWORD",
            },
        },
        "business_glossary": {
            "Revenue": "SUM(sales.amount)",
        },
        "business_rules": [
            "Always exclude test records where is_test = true.",
        ],
        "sensitivity": ["email", "ssn", "credit_score"],
        "row_limits": {"default": 100, "max": 500},
        "few_shot_examples": [
            {
                "question": "How many sales were recorded today?",
                "sql": "SELECT COUNT(*) FROM sales WHERE DATE(created_at) = CURRENT_DATE;",
            },
            {
                "question": "What is the total revenue this month?",
                "sql": (
                    "SELECT SUM(amount) AS revenue FROM sales "
                    "WHERE DATE_TRUNC('month', created_at) = DATE_TRUNC('month', CURRENT_DATE);"
                ),
            },
        ],
        "tables": [
            {
                "name": "sales",
                "description": "Sales transactions",
                "columns": [
                    {"name": "sale_id",    "type": "INTEGER", "primary_key": True},
                    {"name": "amount",     "type": "NUMERIC(12,2)"},
                    {"name": "created_at", "type": "TIMESTAMPTZ"},
                    {"name": "is_test",    "type": "BOOLEAN"},
                ],
            }
        ],
    }


@pytest.fixture()
def mock_store(valid_manifest: dict) -> AsyncMock:
    store = AsyncMock(spec=ManifestStore)
    store.get_manifest.return_value = valid_manifest
    return store


# ── ManifestLoader: happy path ────────────────────────────────────────────────

class TestManifestLoaderValid:
    @pytest.mark.asyncio
    async def test_valid_manifest_loads_correctly(
        self, mock_store: AsyncMock, valid_manifest: dict
    ) -> None:
        loader = ManifestLoader(mock_store)
        result = await loader.load("test")

        assert result == valid_manifest
        mock_store.get_manifest.assert_awaited_once_with("test")

    @pytest.mark.asyncio
    async def test_load_returns_all_required_sections(
        self, mock_store: AsyncMock
    ) -> None:
        loader = ManifestLoader(mock_store)
        result = await loader.load("test")

        for section in REQUIRED_SECTIONS:
            assert section in result, f"Section '{section}' missing from returned manifest"

    @pytest.mark.asyncio
    async def test_domain_id_forwarded_to_store(self, mock_store: AsyncMock) -> None:
        loader = ManifestLoader(mock_store)
        await loader.load("sales")
        mock_store.get_manifest.assert_awaited_once_with("sales")


# ── ManifestLoader: missing sections ─────────────────────────────────────────

class TestManifestLoaderMissingSections:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_section", REQUIRED_SECTIONS)
    async def test_each_missing_section_raises(
        self, mock_store: AsyncMock, valid_manifest: dict, missing_section: str
    ) -> None:
        """Removing any single required section must raise ManifestValidationError."""
        incomplete = {k: v for k, v in valid_manifest.items() if k != missing_section}
        mock_store.get_manifest.return_value = incomplete

        loader = ManifestLoader(mock_store)
        with pytest.raises(ManifestValidationError) as exc_info:
            await loader.load("test")

        err = exc_info.value
        assert err.domain == "test"
        assert missing_section in err.missing

    @pytest.mark.asyncio
    async def test_multiple_missing_sections_all_reported(
        self, mock_store: AsyncMock
    ) -> None:
        """A nearly-empty manifest should list every missing section."""
        mock_store.get_manifest.return_value = {
            "domain": "test",
            "tables": [],          # only `tables` present; all others absent
        }
        loader = ManifestLoader(mock_store)
        with pytest.raises(ManifestValidationError) as exc_info:
            await loader.load("test")

        missing = exc_info.value.missing
        expected_missing = [s for s in REQUIRED_SECTIONS if s != "tables"]
        for section in expected_missing:
            assert section in missing, f"Expected '{section}' in missing list"

    @pytest.mark.asyncio
    async def test_error_message_contains_domain(
        self, mock_store: AsyncMock, valid_manifest: dict
    ) -> None:
        incomplete = {k: v for k, v in valid_manifest.items() if k != "persona"}
        mock_store.get_manifest.return_value = incomplete

        loader = ManifestLoader(mock_store)
        with pytest.raises(ManifestValidationError) as exc_info:
            await loader.load("my_domain")

        assert "my_domain" in str(exc_info.value)


# ── ManifestLoader: wrong types ───────────────────────────────────────────────

class TestManifestLoaderWrongTypes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "section, bad_value",
        [
            ("persona",             "not a dict"),
            ("database_connection", ["list", "not", "dict"]),
            ("business_glossary",   ["list", "not", "dict"]),
            ("business_rules",      {"not": "a list"}),
            ("sensitivity",         "not a list"),
            ("row_limits",          [1, 2, 3]),
            ("few_shot_examples",   {"not": "a list"}),
            ("tables",              "not a list"),
        ],
    )
    async def test_wrong_type_raises_manifest_validation_error(
        self,
        mock_store: AsyncMock,
        valid_manifest: dict,
        section: str,
        bad_value: object,
    ) -> None:
        bad_manifest = dict(valid_manifest)
        bad_manifest[section] = bad_value
        mock_store.get_manifest.return_value = bad_manifest

        loader = ManifestLoader(mock_store)
        with pytest.raises(ManifestValidationError):
            await loader.load("test")


# ── ManifestLoader: store errors propagate ───────────────────────────────────

class TestManifestLoaderStoreErrors:
    @pytest.mark.asyncio
    async def test_not_found_error_propagates(self, mock_store: AsyncMock) -> None:
        mock_store.get_manifest.side_effect = ManifestNotFoundError(
            "Domain 'unknown' not found"
        )
        loader = ManifestLoader(mock_store)
        with pytest.raises(ManifestNotFoundError):
            await loader.load("unknown")

    @pytest.mark.asyncio
    async def test_unexpected_store_error_propagates(self, mock_store: AsyncMock) -> None:
        mock_store.get_manifest.side_effect = RuntimeError("S3 unreachable")
        loader = ManifestLoader(mock_store)
        with pytest.raises(RuntimeError, match="S3 unreachable"):
            await loader.load("sales")


# ── resolve_db_connection ─────────────────────────────────────────────────────

class TestResolveDbConnection:
    def test_resolves_all_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_DB_HOST",     "db.prod.example.com")
        monkeypatch.setenv("TEST_DB_PORT",     "5432")
        monkeypatch.setenv("TEST_DB_NAME",     "salesdb")
        monkeypatch.setenv("TEST_DB_USER",     "readonly_user")
        monkeypatch.setenv("TEST_DB_PASSWORD", "s3cr3t")

        block = {
            "type": "postgresql",
            "env_vars": {
                "host":     "TEST_DB_HOST",
                "port":     "TEST_DB_PORT",
                "database": "TEST_DB_NAME",
                "user":     "TEST_DB_USER",
                "password": "TEST_DB_PASSWORD",
            },
        }
        result = resolve_db_connection(block)

        assert result["host"]     == "db.prod.example.com"
        assert result["port"]     == "5432"
        assert result["database"] == "salesdb"
        assert result["user"]     == "readonly_user"
        assert result["password"] == "s3cr3t"
        assert result["type"]     == "postgresql"

    def test_missing_env_var_raises_environment_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MISSING_DB_VAR", raising=False)
        block = {
            "type": "postgresql",
            "env_vars": {"host": "MISSING_DB_VAR"},
        }
        with pytest.raises(EnvironmentError) as exc_info:
            resolve_db_connection(block)
        assert "MISSING_DB_VAR" in str(exc_info.value)

    def test_multiple_missing_vars_all_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in ("DB_HOST_GONE", "DB_PASS_GONE"):
            monkeypatch.delenv(var, raising=False)

        block = {
            "type": "postgresql",
            "env_vars": {"host": "DB_HOST_GONE", "password": "DB_PASS_GONE"},
        }
        with pytest.raises(EnvironmentError) as exc_info:
            resolve_db_connection(block)

        msg = str(exc_info.value)
        assert "DB_HOST_GONE" in msg
        assert "DB_PASS_GONE" in msg

    def test_default_type_is_postgresql(self) -> None:
        """If 'type' key is absent, default to 'postgresql'."""
        block = {"env_vars": {}}
        result = resolve_db_connection(block)
        assert result["type"] == "postgresql"

    def test_custom_db_type_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_HOST", "localhost")
        block = {
            "type": "mysql",
            "env_vars": {"host": "MY_HOST"},
        }
        result = resolve_db_connection(block)
        assert result["type"] == "mysql"

    def test_empty_env_vars_resolves_only_type(self) -> None:
        block = {"type": "sqlite", "env_vars": {}}
        result = resolve_db_connection(block)
        assert result == {"type": "sqlite"}

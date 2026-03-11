"""
Manifest store
──────────────
Loads domain YAML files from either the local file system or an S3 bucket.
The backend is selected by the MANIFEST_STORE_TYPE environment variable:

    MANIFEST_STORE_TYPE=local   → reads from MANIFEST_LOCAL_PATH/
    MANIFEST_STORE_TYPE=s3      → reads from s3://S3_BUCKET_NAME/S3_MANIFEST_PREFIX

All public methods are async-safe.  The S3 calls are executed in a thread-pool
executor to avoid blocking the event loop (boto3 is synchronous).
"""
import asyncio
import logging
from functools import lru_cache
from pathlib import Path

import yaml

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ManifestNotFoundError(Exception):
    """Raised when a domain manifest cannot be located in the store."""


class ManifestStore:
    """Thin async abstraction over local-FS or S3 YAML manifest files."""

    def __init__(self) -> None:
        self._store_type = settings.manifest_store_type

    # ── Public API ────────────────────────────────────────────────────────────

    async def get_manifest(self, domain: str) -> dict:
        """Return the parsed YAML dict for *domain*."""
        if self._store_type == "s3":
            return await self._get_from_s3(domain)
        return await self._get_from_local(domain)

    async def list_domains(self) -> list[str]:
        """Return all available domain names (file stems)."""
        if self._store_type == "s3":
            return await self._list_s3_domains()
        return await self._list_local_domains()

    async def put_manifest(self, domain: str, data: dict) -> None:
        """Persist *data* as the YAML manifest for *domain* (local only for now)."""
        if self._store_type == "s3":
            raise NotImplementedError("S3 write not yet implemented – upload via AWS CLI or SDK.")
        await self._put_local(domain, data)

    # ── Local ─────────────────────────────────────────────────────────────────

    async def _get_from_local(self, domain: str) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_read_local, domain)

    def _sync_read_local(self, domain: str) -> dict:
        path = Path(settings.manifest_local_path) / f"{domain}.yaml"
        if not path.exists():
            raise ManifestNotFoundError(
                f"Manifest '{domain}.yaml' not found in '{settings.manifest_local_path}'"
            )
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    async def _list_local_domains(self) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_list_local)

    def _sync_list_local(self) -> list[str]:
        base = Path(settings.manifest_local_path)
        if not base.is_dir():
            return []
        return sorted(p.stem for p in base.glob("*.yaml"))

    async def _put_local(self, domain: str, data: dict) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_write_local, domain, data)

    def _sync_write_local(self, domain: str, data: dict) -> None:
        base = Path(settings.manifest_local_path)
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{domain}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
        logger.info("Manifest written: %s", path)

    # ── S3 ────────────────────────────────────────────────────────────────────

    def _s3_client(self):
        import boto3  # imported lazily – not required for local mode

        return boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id or None,
            aws_secret_access_key=settings.aws_secret_access_key or None,
            region_name=settings.aws_region,
        )

    async def _get_from_s3(self, domain: str) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_read_s3, domain)

    def _sync_read_s3(self, domain: str) -> dict:
        key = f"{settings.s3_manifest_prefix}{domain}.yaml"
        client = self._s3_client()
        try:
            response = client.get_object(Bucket=settings.s3_bucket_name, Key=key)
        except client.exceptions.NoSuchKey:  # type: ignore[attr-defined]
            raise ManifestNotFoundError(
                f"Manifest not found in s3://{settings.s3_bucket_name}/{key}"
            )
        content: str = response["Body"].read().decode("utf-8")
        return yaml.safe_load(content)

    async def _list_s3_domains(self) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._sync_list_s3)

    def _sync_list_s3(self) -> list[str]:
        client = self._s3_client()
        paginator = client.get_paginator("list_objects_v2")
        prefix = settings.s3_manifest_prefix
        domains: list[str] = []
        for page in paginator.paginate(Bucket=settings.s3_bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.endswith(".yaml"):
                    stem = key[len(prefix):].removesuffix(".yaml")
                    domains.append(stem)
        return sorted(domains)


# ── Singleton helper ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_manifest_store() -> ManifestStore:
    return ManifestStore()

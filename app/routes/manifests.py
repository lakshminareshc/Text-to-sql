"""
Routes: /api/v1/manifests
"""
import logging

from fastapi import APIRouter, HTTPException

from app.manifests.manifest_store import ManifestNotFoundError, get_manifest_store
from app.models.schemas import ManifestDetail, ManifestSummary

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/manifests",
    response_model=list[ManifestSummary],
    summary="List available domain manifests",
)
async def list_manifests() -> list[ManifestSummary]:
    store = get_manifest_store()
    domains = await store.list_domains()
    summaries: list[ManifestSummary] = []
    for domain in domains:
        try:
            manifest = await store.get_manifest(domain)
            summaries.append(
                ManifestSummary(
                    domain=domain,
                    description=manifest.get("description"),
                    table_count=len(manifest.get("tables", [])),
                )
            )
        except Exception as exc:
            logger.warning("Could not load manifest '%s': %s", domain, exc)
    return summaries


@router.get(
    "/manifests/{domain}",
    response_model=ManifestDetail,
    summary="Get raw manifest for a domain",
)
async def get_manifest(domain: str) -> ManifestDetail:
    store = get_manifest_store()
    try:
        manifest = await store.get_manifest(domain)
    except ManifestNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error loading manifest '%s': %s", domain, exc)
        raise HTTPException(status_code=500, detail="Failed to load manifest.") from exc
    return ManifestDetail(domain=domain, manifest=manifest)

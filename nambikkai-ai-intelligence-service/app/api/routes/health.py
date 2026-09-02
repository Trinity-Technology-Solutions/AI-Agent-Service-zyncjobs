from fastapi import APIRouter
from app.core.config import get_settings
from app.providers import get_provider

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "nambikkai-ai-intelligence-service"}


@router.get("/ready")
async def ready():
    settings = get_settings()
    provider = get_provider()
    provider_reachable = await provider.health_check()
    return {
        "status": "ready" if provider_reachable else "degraded",
        "provider": settings.AI_PROVIDER,
        "provider_reachable": provider_reachable,
    }

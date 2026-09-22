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


@router.get("/xgboost-status")
async def xgboost_status():
    """
    Return the complete real-time XGBoost pipeline status.

    This is the single authoritative source of truth for:
    - data_prerequisites_met (from live DB readiness check)
    - model_trained (artifact exists on disk)
    - model_loaded (artifact parsed and in memory)
    - prediction_available (loaded AND passed quality gates)
    - qualification_status and metrics

    The dashboard must consume this endpoint — never hardcode these states.
    """
    from app.ml.readiness import get_xgboost_status_detail
    status = get_xgboost_status_detail()
    return {"ok": True, "xgboost": status}


@router.get("/email/status")
async def email_status():
    from app.services.email_reporter import get_email_status
    return {"ok": True, "email": get_email_status()}


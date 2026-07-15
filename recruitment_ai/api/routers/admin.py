"""Admin endpoints — system monitoring, analytics, brain stats."""
import logging
from fastapi import APIRouter, Depends
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.config.settings import settings
from recruitment_ai.brains.master.master_brain import master_brain
from recruitment_ai.vector.store import vector_store
from recruitment_ai.shared.llm_service import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/admin", tags=["admin"])


@router.get("/health")
async def admin_health(user: dict = Depends(get_current_user)):
    """Detailed health check for all system components."""
    import time
    results = {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": time.time(),
        "checks": {},
    }
    try:
        llm_ok = await llm_service.health_check()
        results["checks"]["llm"] = "ok" if llm_ok else "degraded"
    except Exception as e:
        results["checks"]["llm"] = f"error: {e}"

    results["checks"]["vector_store"] = f"{vector_store.count} chunks" if vector_store.count > 0 else "empty"
    results["checks"]["brains"] = f"{len(master_brain.brains)} registered"
    results["checks"]["provider"] = settings.LLM_PROVIDER

    all_ok = all(v == "ok" or "registered" in str(v) or "chunks" in str(v) for v in results["checks"].values())
    results["status"] = "healthy" if all_ok else "degraded"
    return results


@router.get("/brains")
async def brain_stats(user: dict = Depends(get_current_user)):
    """List all registered brains with their intents."""
    brains = {}
    for intent, brain in master_brain.brains.items():
        brains[intent] = {
            "name": getattr(brain, "name", brain.__class__.__name__),
            "type": brain.__class__.__name__,
        }
    return {
        "total": len(brains),
        "provider": settings.LLM_PROVIDER,
        "model": settings._DEFAULT_MODEL,
        "brains": brains,
    }


@router.get("/config")
async def admin_config(user: dict = Depends(get_current_user)):
    """Show sanitized system configuration."""
    return {
        "environment": settings.ENVIRONMENT,
        "llm_provider": settings.LLM_PROVIDER,
        "vector_backend": getattr(vector_store, "backend", "unknown"),
        "qdrant_url": settings.QDRANT_URL,
        "chromadb_path": settings.CHROMADB_PATH,
        "redis_url": f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
        "rag_top_k": settings.RAG_TOP_K,
        "rate_limit": f"{settings.RATE_LIMIT_REQUESTS}/{settings.RATE_LIMIT_WINDOW_SECONDS}s",
        "textract_enabled": settings.TEXTRACT_ENABLED,
        "backend_api": settings.BACKEND_API_URL,
    }


@router.get("/metrics/summary")
async def metrics_summary(user: dict = Depends(get_current_user)):
    """Summary of key metrics (prometheus data accessible at /metrics)."""
    from prometheus_client import REGISTRY
    summary = {}
    for name in ["zyncjobs_ai_requests_total", "zyncjobs_ai_latency_seconds", "zyncjobs_cache_hits_total"]:
        try:
            metric = REGISTRY.get_sample_value(name) if name in [s.name for s in REGISTRY.collect()] else 0
            summary[name] = metric
        except Exception:
            summary[name] = "unavailable"
    return summary

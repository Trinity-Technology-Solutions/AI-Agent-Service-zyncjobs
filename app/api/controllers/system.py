"""System health, version, and metrics endpoints."""
import httpx
from fastapi import APIRouter
from app.config.settings import settings
from app.gateway.service_registry import service_registry
from app.knowledge.knowledge_base import knowledge_base
from app.memory.memory_manager import memory
from app.memory.cache import prompt_cache
from app.metrics.collector import metrics_collector

router = APIRouter(tags=["System"])


@router.get("/health")
async def health():
    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=3)
            ollama_status = "connected" if resp.status_code == 200 else "error"
    except Exception:
        ollama_status = "unreachable"

    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "ollama": ollama_status,
        "model": settings.OLLAMA_MODEL,
        "memory": "ok" if memory else "error",
        "knowledge": f"{knowledge_base.document_count} docs loaded" if knowledge_base.document_count else "empty",
        "services": len(service_registry.list()),
        "cache": prompt_cache.size,
    }


@router.get("/version")
def version():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "llm": "ollama",
        "model": settings.OLLAMA_MODEL,
        "services": service_registry.list_with_info(),
    }


@router.get("/metrics")
def metrics():
    return metrics_collector.summary()

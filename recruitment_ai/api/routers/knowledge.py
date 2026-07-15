"""Knowledge base endpoints — search, stats, ingest."""
import logging
from fastapi import APIRouter, Depends, Query
from typing import Optional
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.vector.store import vector_store
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/knowledge", tags=["knowledge"])


@router.get("/search")
async def search_knowledge(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(5, description="Number of results"),
    user: dict = Depends(get_current_user),
):
    """Search the knowledge base for relevant documents."""
    try:
        results = await vector_store.search(query, top_k=top_k)
        return {
            "success": True,
            "query": query,
            "total": len(results),
            "results": [
                {
                    "text": getattr(r, "text", str(r))[:500],
                    "title": getattr(r, "metadata", {}).get("title", "") if hasattr(r, "metadata") else "",
                    "url": getattr(r, "metadata", {}).get("url", "") if hasattr(r, "metadata") else "",
                    "score": getattr(r, "score", 0) if hasattr(r, "score") else 0,
                }
                for r in results
            ],
        }
    except Exception as e:
        logger.warning("Knowledge search failed: %s", e)
        return {"success": False, "error": str(e), "results": []}


@router.get("/stats")
async def knowledge_stats(user: dict = Depends(get_current_user)):
    """Get knowledge base statistics."""
    try:
        return {
            "success": True,
            "total_chunks": vector_store.count,
            "collection": settings.QDRANT_COLLECTION,
            "backend": getattr(vector_store, "backend", "unknown"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

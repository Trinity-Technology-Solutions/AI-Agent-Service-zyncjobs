from fastapi import APIRouter, Query
from typing import Optional
from app.models.ai_audit_log import get_logs, get_stats

router = APIRouter()


@router.get("/logs")
def get_audit_logs(
    feature: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
):
    return {
        "logs": get_logs(feature=feature, status=status, limit=limit, offset=offset),
        "total": len(get_logs(feature=feature, status=status, limit=999999)),
        "limit": limit,
        "offset": offset,
    }


@router.get("/stats")
def get_audit_stats():
    return get_stats()


@router.get("/features")
def get_audit_features():
    stats = get_stats()
    return {"features": stats.get("features", [])}

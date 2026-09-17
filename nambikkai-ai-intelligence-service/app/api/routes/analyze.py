"""
POST /analyze — real-data analysis endpoint.

Accepts a platform + content_id, fetches real analytics from PostgreSQL,
runs the full deterministic + LLM pipeline, and returns the result.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.exceptions import DataSourceConnectionError, DataSourceError
from app.core.logging import get_logger
from app.services.analyze_service import run_real_analysis

router = APIRouter()
logger = get_logger(__name__)

_SUPPORTED_PLATFORMS = {"youtube", "instagram", "facebook"}


class AnalyzeRequest(BaseModel):
    platform: str
    content_id: str


@router.post("/analyze")
async def analyze(request: AnalyzeRequest):
    platform = request.platform.lower().strip()
    content_id = request.content_id.strip()

    if platform not in _SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported platform '{platform}'. Supported: {sorted(_SUPPORTED_PLATFORMS)}",
        )

    if not content_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="content_id must not be empty",
        )

    try:
        result, audit, coverage, evidence = await run_real_analysis(platform, content_id)
    except DataSourceConnectionError as exc:
        logger.error("Database connection error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        )
    except DataSourceError as exc:
        logger.error("Data source error for %s/%s: %s", platform, content_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Content not found or query failed: {content_id}",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    return {
        "platform": platform,
        "content_id": content_id,
        "analysis": result.model_dump(),
        "audit": audit.model_dump(),
        "baseline_coverage": coverage.model_dump(),
        "evidence": evidence.model_dump(),
    }

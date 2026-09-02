"""
INTERNAL DEVELOPMENT ENDPOINT — NOT FOR PUBLIC/PRODUCTION USE.

This endpoint accepts controlled test data and runs the full agent pipeline.
It is intended for local development and integration testing only.
"""
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.orchestrator import AgentOrchestrator
from app.domain.models import (
    AnalyticsEvent,
    ContentMetadata,
    ContentMetrics,
)
from app.providers import get_provider

router = APIRouter()


class TestAnalysisRequest(BaseModel):
    event_id: str = "test-001"
    content_id: str = "content-001"
    title: str = "Test Content"
    creator_id: str = "creator-001"
    platform: str | None = "youtube"
    current_hour_delta_views: float = 0.0
    seven_day_rolling_hourly_baseline: float = 0.0
    one_hour_delta_likes: float = 0.0
    one_hour_delta_views: float = 0.0


@router.post("/internal/test-analysis")
async def test_analysis(request: TestAnalysisRequest):
    event = AnalyticsEvent(
        event_id=request.event_id,
        received_at=datetime.now(timezone.utc),
        metadata=ContentMetadata(
            content_id=request.content_id,
            title=request.title,
            creator_id=request.creator_id,
            platform=request.platform,
        ),
        metrics=ContentMetrics(
            current_hour_delta_views=request.current_hour_delta_views,
            seven_day_rolling_hourly_baseline=request.seven_day_rolling_hourly_baseline,
            one_hour_delta_likes=request.one_hour_delta_likes,
            one_hour_delta_views=request.one_hour_delta_views,
        ),
    )
    provider = get_provider()
    orchestrator = AgentOrchestrator(provider=provider)
    result, audit = await orchestrator.run(event)
    return {
        "analysis": result.model_dump(),
        "audit": audit.model_dump(),
    }

from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────

class GateClassification(str, Enum):
    NOMINAL = "NOMINAL"
    ELEVATED = "ELEVATED"
    BOOMING_SURGE = "BOOMING_SURGE"


class AnalysisStatus(str, Enum):
    SUCCESS = "SUCCESS"
    SKIPPED_NOMINAL = "SKIPPED_NOMINAL"
    MONITORING = "MONITORING"
    INVALID = "INVALID"
    PROVIDER_ERROR = "PROVIDER_ERROR"


# ── Input domain ───────────────────────────────────────────────────────────

class ContentMetadata(BaseModel):
    content_id: str
    title: str
    creator_id: str
    platform: Optional[str] = None
    published_at: Optional[datetime] = None


class ContentMetrics(BaseModel):
    current_hour_delta_views: float = Field(ge=0)
    seven_day_rolling_hourly_baseline: float = Field(ge=0)
    one_hour_delta_likes: float = Field(ge=0)
    one_hour_delta_views: float = Field(ge=0)
    total_views: Optional[float] = None
    total_likes: Optional[float] = None


class AnalyticsEvent(BaseModel):
    event_id: str
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: ContentMetadata
    metrics: ContentMetrics


# ── Gating ─────────────────────────────────────────────────────────────────

class GateResult(BaseModel):
    classification: GateClassification
    velocity_ratio: Optional[float] = None
    like_acceleration: Optional[float] = None
    reason: str


# ── Evidence ───────────────────────────────────────────────────────────────

class DataQuality(BaseModel):
    baseline_available: bool = True
    metrics_complete: bool = True
    notes: Optional[str] = None


class EvidencePackage(BaseModel):
    content_metadata: ContentMetadata
    verified_metrics: ContentMetrics
    gate_result: GateResult
    transcript_excerpt: Optional[str] = None
    regional_signals: Optional[dict] = None
    data_quality: DataQuality = Field(default_factory=DataQuality)


# ── AI output ──────────────────────────────────────────────────────────────

class EditorialAnalysis(BaseModel):
    content_intent: str
    observed_signals: list[str]
    possible_contributing_factors: list[str]
    writer_recommendations: list[str]
    keyword_suggestions: list[str]
    title_suggestions: list[str]
    description_suggestions: list[str]
    hashtag_suggestions: list[str]
    cross_platform_ideas: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str]


# ── Validation ─────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    is_valid: bool
    failures: list[str] = Field(default_factory=list)


# ── Final result ───────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    status: AnalysisStatus
    gate_result: GateResult
    editorial_analysis: Optional[EditorialAnalysis] = None
    validation_result: Optional[ValidationResult] = None
    message: Optional[str] = None


# ── Audit ──────────────────────────────────────────────────────────────────

class AuditRecord(BaseModel):
    event_id: str
    content_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    gate_classification: GateClassification
    analysis_status: AnalysisStatus
    provider_used: Optional[str] = None
    validation_passed: Optional[bool] = None
    llm_invoked: bool = False

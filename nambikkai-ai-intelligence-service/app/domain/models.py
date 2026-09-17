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
    SURGE_CANDIDATE = "SURGE_CANDIDATE"
    LOW_PERFORMING = "LOW_PERFORMING"


class AnalysisStatus(str, Enum):
    SUCCESS = "SUCCESS"
    SKIPPED_NOMINAL = "SKIPPED_NOMINAL"
    MONITORING = "MONITORING"
    INVALID = "INVALID"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    WAITING_FOR_DATA = "WAITING_FOR_DATA"


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


# ── PostgreSQL normalized representation ───────────────────────────────────

class HistorySnapshot(BaseModel):
    """One row from a *_history table, platform-agnostic."""
    collected_at: datetime
    published_at: datetime
    primary_metric_name: str   # "views" | "reach"
    primary_metric_value: int
    likes: int
    comments: int


class NormalizedContentRecord(BaseModel):
    """
    Joined result of *_history + *_content for a single content item.
    Preserves platform-specific metric semantics.
    """
    platform: str
    content_id: str
    account_key: str           # channel_key | account_key
    primary_metric_name: str   # "views" for YouTube, "reach" for Instagram/Facebook
    history: list[HistorySnapshot]
    # Content metadata — None when the content row is missing
    title: Optional[str] = None          # YouTube only
    caption: Optional[str] = None        # Instagram / Facebook
    description: Optional[str] = None    # YouTube only
    category: Optional[str] = None
    language: Optional[str] = None
    creator_id: Optional[str] = None     # YouTube only
    media_type: Optional[str] = None     # Instagram only
    post_type: Optional[str] = None      # Facebook only
    url: Optional[str] = None
    content_published_at: Optional[datetime] = None


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
    llm_eligible: bool = False
    raw_classification: Optional[GateClassification] = None


# ── Baseline coverage ─────────────────────────────────────────────────────

class BaselineCoverage(BaseModel):
    """
    Describes how much history was actually available versus what was requested.

    When available_hours < requested_hours the baseline is computed from a
    shorter window than intended.  The gating layer uses this to prevent an
    insufficient baseline from silently producing a high-confidence surge.
    """
    requested_hours: float
    available_hours: float
    sufficient: bool  # True only when available_hours >= requested_hours


# ── Evidence ───────────────────────────────────────────────────────────────

class DataQuality(BaseModel):
    baseline_available: bool = True
    metrics_complete: bool = True
    notes: Optional[str] = None


class EvidencePackage(BaseModel):
    content_metadata: ContentMetadata
    verified_metrics: ContentMetrics
    gate_result: GateResult
    baseline_coverage: Optional[BaselineCoverage] = None
    transcript_excerpt: Optional[str] = None
    regional_signals: Optional[dict] = None
    data_quality: DataQuality = Field(default_factory=DataQuality)


# ── AI output ──────────────────────────────────────────────────────────────

class EditorialAnalysis(BaseModel):
    content_intent: str
    observed_signals: list[str]
    possible_contributing_factors: list[str]
    writer_recommendations: list[str]
    keyword_suggestions: list[str] = Field(default_factory=list)
    title_suggestions: list[str] = Field(default_factory=list)
    description_suggestions: list[str] = Field(default_factory=list)
    hashtag_suggestions: list[str] = Field(default_factory=list)
    cross_platform_ideas: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    publishing_timing: Optional[str] = None



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
    raw_classification: Optional[GateClassification] = None

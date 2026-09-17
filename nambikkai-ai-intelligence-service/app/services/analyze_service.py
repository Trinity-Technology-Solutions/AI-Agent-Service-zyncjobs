"""
Real-data analysis service.

Orchestrates the complete pipeline:
  PostgreSQL -> NormalizedContentRecord -> ContentMetrics + BaselineCoverage
  -> evaluate_gate_with_coverage -> build_evidence_package -> AgentOrchestrator

Returns (AnalysisResult, AuditRecord, BaselineCoverage, EvidencePackage) so
callers can inspect the exact evidence that was constructed for the LLM.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.agent.orchestrator import AgentOrchestrator
from app.audit.models import AuditRecord
from app.core.exceptions import DataSourceConnectionError, DataSourceError
from app.data_sources.postgres import fetch_normalized_record
from app.domain.models import (
    AnalysisResult,
    AnalysisStatus,
    AnalyticsEvent,
    BaselineCoverage,
    ContentMetadata,
    EvidencePackage,
    GateClassification,
    GateResult,
)
from app.providers import get_provider
from app.services.evidence_builder import build_evidence_package
from app.services.gating import evaluate_gate_with_coverage
from app.services.metrics_from_history import (
    DEFAULT_BASELINE_HOURS,
    metrics_and_coverage_from_history,
)

_HISTORY_FETCH_HOURS: int = 720  # 30 days; returns whatever actually exists


async def run_real_analysis(
    platform: str,
    content_id: str,
) -> tuple[AnalysisResult, AuditRecord, BaselineCoverage, EvidencePackage]:
    """
    Execute the full real-data analysis pipeline for one content item.

    Returns:
        (AnalysisResult, AuditRecord, BaselineCoverage, EvidencePackage)

    The EvidencePackage is the exact object constructed from real DB data,
    including the coverage note in data_quality.notes.

    Raises:
        ValueError: unsupported platform.
        DataSourceConnectionError: database pool not initialised.
        DataSourceError: query failed or content_id not found.
    """
    # ── 1. Fetch real data from PostgreSQL ────────────────────────────────
    record = await fetch_normalized_record(platform, content_id, hours=_HISTORY_FETCH_HOURS)

    if not record.history:
        gate = GateResult(
            classification=GateClassification.NOMINAL,
            reason="No history records available for this content item",
        )
        coverage = BaselineCoverage(
            requested_hours=DEFAULT_BASELINE_HOURS,
            available_hours=0.0,
            sufficient=False,
        )
        # Build a minimal evidence package for the no-history case
        title = record.title or record.caption or f"{platform}/{content_id}"
        creator = record.creator_id or record.account_key or "unknown"
        from app.domain.models import ContentMetrics
        empty_metrics = ContentMetrics(
            current_hour_delta_views=0.0,
            seven_day_rolling_hourly_baseline=0.0,
            one_hour_delta_likes=0.0,
            one_hour_delta_views=0.0,
        )
        metadata = ContentMetadata(
            content_id=content_id,
            title=title,
            creator_id=creator,
            platform=platform,
            published_at=record.content_published_at,
        )
        evidence = build_evidence_package(metadata=metadata, metrics=empty_metrics, gate_result=gate)
        evidence.data_quality.notes = "No history data available"
        result = AnalysisResult(
            status=AnalysisStatus.SKIPPED_NOMINAL,
            gate_result=gate,
            message=f"No history data for {platform}/{content_id}",
        )
        audit = AuditRecord(
            event_id=str(uuid.uuid4()),
            content_id=content_id,
            gate_classification=gate.classification,
            analysis_status=result.status,
            llm_invoked=False,
        )
        return result, audit, coverage, evidence

    # ── 2. Deterministic metrics + baseline coverage ──────────────────────
    metrics, coverage = metrics_and_coverage_from_history(
        record, requested_baseline_hours=DEFAULT_BASELINE_HOURS
    )

    # ── 3. Coverage-aware gate ────────────────────────────────────────────
    gate = evaluate_gate_with_coverage(metrics, coverage)

    # ── 4. Content metadata ───────────────────────────────────────────────
    title = record.title or record.caption or f"{platform}/{content_id}"
    creator = record.creator_id or record.account_key or "unknown"
    metadata = ContentMetadata(
        content_id=content_id,
        title=title,
        creator_id=creator,
        platform=platform,
        published_at=record.content_published_at,
    )

    # ── 5. Evidence with structured coverage ─────────────────────────────
    evidence = build_evidence_package(
        metadata=metadata,
        metrics=metrics,
        gate_result=gate,
        coverage=coverage,
    )

    # ── 6. Orchestrator ───────────────────────────────────────────────────
    event = AnalyticsEvent(
        event_id=str(uuid.uuid4()),
        received_at=datetime.now(timezone.utc),
        metadata=metadata,
        metrics=metrics,
    )
    provider = get_provider()
    orchestrator = AgentOrchestrator(provider=provider)
    # Pass the coverage-aware gate and pre-built evidence so the orchestrator
    # does not re-evaluate the gate (which would lose SURGE_CANDIDATE).
    result, audit = await orchestrator.run(event, gate_result=gate, evidence=evidence)

    return result, audit, coverage, evidence

from app.domain.models import (
    BaselineCoverage,
    ContentMetadata,
    ContentMetrics,
    DataQuality,
    EvidencePackage,
    GateClassification,
    GateResult,
)


def build_evidence_package(
    metadata: ContentMetadata,
    metrics: ContentMetrics,
    gate_result: GateResult,
    coverage: BaselineCoverage | None = None,
    transcript_excerpt: str | None = None,
    regional_signals: dict | None = None,
) -> EvidencePackage:
    quality = DataQuality(
        baseline_available=metrics.seven_day_rolling_hourly_baseline > 0,
        metrics_complete=all([
            metrics.current_hour_delta_views >= 0,
            metrics.one_hour_delta_likes >= 0,
            metrics.one_hour_delta_views >= 0,
        ]),
    )
    if coverage is not None:
        status = "sufficient" if coverage.sufficient else "INSUFFICIENT"
        quality.notes = (
            f"Baseline: {coverage.available_hours:.1f}h available / "
            f"{coverage.requested_hours:.0f}h requested — {status}"
        )
        if not coverage.sufficient and gate_result.classification == GateClassification.SURGE_CANDIDATE:
            quality.notes += (
                "; SURGE_CANDIDATE: early signal only — "
                "insufficient history to confirm sustained trend"
            )
    return EvidencePackage(
        content_metadata=metadata,
        verified_metrics=metrics,
        gate_result=gate_result,
        baseline_coverage=coverage,
        transcript_excerpt=transcript_excerpt[:500] if transcript_excerpt else None,
        regional_signals=regional_signals,
        data_quality=quality,
    )

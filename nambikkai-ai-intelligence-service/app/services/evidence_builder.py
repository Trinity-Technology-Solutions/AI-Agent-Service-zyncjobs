from app.domain.models import (
    ContentMetadata,
    ContentMetrics,
    DataQuality,
    EvidencePackage,
    GateResult,
)


def build_evidence_package(
    metadata: ContentMetadata,
    metrics: ContentMetrics,
    gate_result: GateResult,
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
    return EvidencePackage(
        content_metadata=metadata,
        verified_metrics=metrics,
        gate_result=gate_result,
        transcript_excerpt=transcript_excerpt[:500] if transcript_excerpt else None,
        regional_signals=regional_signals,
        data_quality=quality,
    )

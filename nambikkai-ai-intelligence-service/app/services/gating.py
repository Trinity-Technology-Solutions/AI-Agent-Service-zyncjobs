from app.core.config import get_settings
from app.domain.models import ContentMetrics, GateClassification, GateResult


def compute_velocity_ratio(metrics: ContentMetrics) -> float | None:
    """current_hour_delta_views / seven_day_rolling_hourly_baseline. None if baseline is zero."""
    if metrics.seven_day_rolling_hourly_baseline == 0:
        return None
    return metrics.current_hour_delta_views / metrics.seven_day_rolling_hourly_baseline


def compute_like_acceleration(metrics: ContentMetrics) -> float | None:
    """(one_hour_delta_likes / one_hour_delta_views) × 100. None if views delta is zero."""
    if metrics.one_hour_delta_views == 0:
        return None
    return (metrics.one_hour_delta_likes / metrics.one_hour_delta_views) * 100


def evaluate_gate(metrics: ContentMetrics) -> GateResult:
    settings = get_settings()
    velocity = compute_velocity_ratio(metrics)
    acceleration = compute_like_acceleration(metrics)

    if velocity is None:
        return GateResult(
            classification=GateClassification.NOMINAL,
            velocity_ratio=None,
            like_acceleration=acceleration,
            reason="Baseline unavailable — defaulting to NOMINAL",
        )

    if velocity >= settings.VELOCITY_SURGE_THRESHOLD:
        return GateResult(
            classification=GateClassification.BOOMING_SURGE,
            velocity_ratio=velocity,
            like_acceleration=acceleration,
            reason=f"Velocity ratio {velocity:.2f} ≥ surge threshold {settings.VELOCITY_SURGE_THRESHOLD}",
        )

    if velocity >= settings.VELOCITY_NOMINAL_THRESHOLD:
        return GateResult(
            classification=GateClassification.ELEVATED,
            velocity_ratio=velocity,
            like_acceleration=acceleration,
            reason=f"Velocity ratio {velocity:.2f} ≥ nominal threshold {settings.VELOCITY_NOMINAL_THRESHOLD}",
        )

    return GateResult(
        classification=GateClassification.NOMINAL,
        velocity_ratio=velocity,
        like_acceleration=acceleration,
        reason=f"Velocity ratio {velocity:.2f} below nominal threshold {settings.VELOCITY_NOMINAL_THRESHOLD}",
    )

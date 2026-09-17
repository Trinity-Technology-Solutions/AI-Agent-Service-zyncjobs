from app.core.config import get_settings
from app.domain.models import BaselineCoverage, ContentMetrics, GateClassification, GateResult


def compute_velocity_ratio(metrics: ContentMetrics) -> float | None:
    """current_hour_delta_views / seven_day_rolling_hourly_baseline. None if baseline is zero."""
    if metrics.seven_day_rolling_hourly_baseline == 0:
        return None
    return metrics.current_hour_delta_views / metrics.seven_day_rolling_hourly_baseline


def compute_like_acceleration(metrics: ContentMetrics) -> float | None:
    """(one_hour_delta_likes / one_hour_delta_views) * 100. None if views delta is zero."""
    if metrics.one_hour_delta_views == 0:
        return None
    return (metrics.one_hour_delta_likes / metrics.one_hour_delta_views) * 100


def build_human_explanation(
    classification: GateClassification,
    velocity_ratio: float | None,
    like_acceleration: float | None,
    coverage: BaselineCoverage | None = None,
) -> str:
    """Generate a clear, CEO-friendly explanation based on deterministic facts without raw formulas."""
    if classification == GateClassification.BOOMING_SURGE:
        return (
            "Confirmed surge pattern: This content is receiving substantially stronger recent view momentum "
            "and engagement than its normal historical baseline, indicating an active audience response."
        )
    elif classification == GateClassification.SURGE_CANDIDATE:
        cov_info = f" ({coverage.available_hours:.1f}h observed)" if coverage else ""
        return (
            f"SURGE_CANDIDATE (Early surge candidate): Content is showing emerging growth velocity above baseline{cov_info}. "
            "Signal is developing; monitor traction and consider promotional support."
        )
    elif classification == GateClassification.ELEVATED:
        return (
            "Above-average engagement: Performance is running moderately above normal channel baseline. "
            "Content shows steady traction and should be monitored for potential acceleration."
        )
    elif classification == GateClassification.LOW_PERFORMING:
        cov_info = f" across {coverage.available_hours:.1f}h of history" if coverage else ""
        return (
            f"LOW_PERFORMING (Sub-baseline performance / low-performing): Content velocity has dropped substantially below its established baseline{cov_info}. "
            "Viewer engagement has stalled; content review, packaging refresh, or re-distribution is recommended."
        )


    return "Nominal performance: Content metrics are tracking within expected baseline variance."



def evaluate_gate(metrics: ContentMetrics) -> GateResult:
    """
    Deterministic surge gate per R&D spec:

      Vr < 1.5                        -> NOMINAL
      1.5 <= Vr < 2.5                 -> ELEVATED
      Vr >= 2.5 OR La >= 8%           -> BOOMING_SURGE (llm_eligible=True)
      Baseline unavailable (Vr=None)  -> NOMINAL (safe default)
    """
    settings = get_settings()
    velocity = compute_velocity_ratio(metrics)
    acceleration = compute_like_acceleration(metrics)

    if velocity is None:
        return GateResult(
            classification=GateClassification.NOMINAL,
            velocity_ratio=None,
            like_acceleration=acceleration,
            reason="Normal baseline performance: Insufficient baseline history to determine velocity ratio.",
            llm_eligible=False,
        )

    # La-triggered surge (even if Vr is below surge threshold)
    la_surge = (
        acceleration is not None
        and acceleration >= settings.LIKE_ACCELERATION_THRESHOLD
    )

    if velocity >= settings.VELOCITY_SURGE_THRESHOLD or la_surge:
        reason = build_human_explanation(GateClassification.BOOMING_SURGE, velocity, acceleration)
        return GateResult(
            classification=GateClassification.BOOMING_SURGE,
            velocity_ratio=velocity,
            like_acceleration=acceleration,
            reason=reason,
            llm_eligible=True,
        )

    if velocity >= settings.VELOCITY_NOMINAL_THRESHOLD:
        reason = build_human_explanation(GateClassification.ELEVATED, velocity, acceleration)
        return GateResult(
            classification=GateClassification.ELEVATED,
            velocity_ratio=velocity,
            like_acceleration=acceleration,
            reason=reason,
            llm_eligible=False,
        )

    reason = build_human_explanation(GateClassification.NOMINAL, velocity, acceleration)
    return GateResult(
        classification=GateClassification.NOMINAL,
        velocity_ratio=velocity,
        like_acceleration=acceleration,
        reason=reason,
        llm_eligible=False,
    )


def evaluate_gate_with_coverage(
    metrics: ContentMetrics,
    coverage: BaselineCoverage,
) -> GateResult:
    """
    Evaluate the gate using real metrics AND baseline coverage information.

    Decision table:
      BOOMING_SURGE + sufficient baseline  -> BOOMING_SURGE  (llm_eligible=True)
      BOOMING_SURGE + insufficient baseline -> SURGE_CANDIDATE (llm_eligible=True)
      ELEVATED (any coverage)              -> ELEVATED       (llm_eligible=False)
      NOMINAL  + sufficient + Vr < low_thr -> LOW_PERFORMING (llm_eligible=True)
      NOMINAL  (any other)                 -> NOMINAL        (llm_eligible=False)
    """
    settings = get_settings()
    gate = evaluate_gate(metrics)

    if not coverage.sufficient and gate.classification == GateClassification.BOOMING_SURGE:
        reason = build_human_explanation(
            GateClassification.SURGE_CANDIDATE,
            gate.velocity_ratio,
            gate.like_acceleration,
            coverage,
        )
        return GateResult(
            classification=GateClassification.SURGE_CANDIDATE,
            raw_classification=GateClassification.BOOMING_SURGE,
            velocity_ratio=gate.velocity_ratio,
            like_acceleration=gate.like_acceleration,
            reason=reason,
            llm_eligible=True,
        )

    # LOW_PERFORMING: only classified when we have a reliable baseline
    if (
        coverage.sufficient
        and gate.classification == GateClassification.NOMINAL
        and gate.velocity_ratio is not None
        and gate.velocity_ratio < settings.LOW_PERFORMING_VELOCITY_THRESHOLD
    ):
        reason = build_human_explanation(
            GateClassification.LOW_PERFORMING,
            gate.velocity_ratio,
            gate.like_acceleration,
            coverage,
        )
        return GateResult(
            classification=GateClassification.LOW_PERFORMING,
            velocity_ratio=gate.velocity_ratio,
            like_acceleration=gate.like_acceleration,
            reason=reason,
            llm_eligible=True,
        )

    return gate



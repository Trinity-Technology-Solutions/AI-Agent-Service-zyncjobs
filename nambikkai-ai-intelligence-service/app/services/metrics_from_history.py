"""
Compute ContentMetrics from a NormalizedContentRecord's real history snapshots.

This is the bridge between the PostgreSQL data layer and the existing
deterministic gating pipeline.  It must NOT call an LLM or make gating
decisions — it only computes the numerical inputs that gating needs.

Metric semantics preserved:
  YouTube    → primary_metric_name = "views"
  Instagram  → primary_metric_name = "reach"
  Facebook   → primary_metric_name = "reach"

The existing ContentMetrics field names use "views" terminology because
the model was designed before multi-platform support.  The mapping is:

  current_hour_delta_views  → delta of primary metric over the most recent interval
  one_hour_delta_views      → same value (alias kept for gating compatibility)
  seven_day_rolling_hourly_baseline → rolling hourly average over all available history
                                      (0.0 when < 2 snapshots exist)
  one_hour_delta_likes      → delta of likes over the most recent interval
  total_views               → latest primary metric absolute value
  total_likes               → latest likes absolute value

If fewer than 2 snapshots exist, baseline is 0.0 (gating will classify NOMINAL,
which is the correct safe default per the existing gating logic).

Baseline sufficiency
--------------------
The requested baseline window (default 168 hours = 7 days) is compared against
the actual span of available history.  When available_hours < requested_hours
the BaselineCoverage.sufficient flag is False.  Callers (e.g. evaluate_gate_with_coverage)
use this to prevent an insufficient baseline from silently producing a
high-confidence BOOMING_SURGE classification.
"""
from __future__ import annotations

from datetime import timedelta

from app.domain.models import BaselineCoverage, ContentMetrics, NormalizedContentRecord

# Default requested baseline window — matches the field name "seven_day_rolling_hourly_baseline"
DEFAULT_BASELINE_HOURS: float = 168.0


def metrics_and_coverage_from_history(
    record: NormalizedContentRecord,
    requested_baseline_hours: float = DEFAULT_BASELINE_HOURS,
) -> tuple[ContentMetrics, BaselineCoverage]:
    """
    Derive ContentMetrics and BaselineCoverage from real history snapshots.

    Returns:
        (ContentMetrics, BaselineCoverage)

    BaselineCoverage.sufficient is True only when the actual history span
    covers at least the requested_baseline_hours window.
    """
    snapshots = record.history

    if not snapshots:
        coverage = BaselineCoverage(
            requested_hours=requested_baseline_hours,
            available_hours=0.0,
            sufficient=False,
        )
        return ContentMetrics(
            current_hour_delta_views=0.0,
            seven_day_rolling_hourly_baseline=0.0,
            one_hour_delta_likes=0.0,
            one_hour_delta_views=0.0,
            total_views=None,
            total_likes=None,
        ), coverage

    latest = snapshots[-1]
    total_metric = float(latest.primary_metric_value)
    total_likes_val = float(latest.likes)

    if len(snapshots) < 2:
        coverage = BaselineCoverage(
            requested_hours=requested_baseline_hours,
            available_hours=0.0,
            sufficient=False,
        )
        return ContentMetrics(
            current_hour_delta_views=0.0,
            seven_day_rolling_hourly_baseline=0.0,
            one_hour_delta_likes=0.0,
            one_hour_delta_views=0.0,
            total_views=total_metric,
            total_likes=total_likes_val,
        ), coverage

    prev = snapshots[-2]
    oldest = snapshots[0]

    # ── Most-recent interval delta ─────────────────────────────────────────
    dt_hours = _delta_hours(prev.collected_at, latest.collected_at)
    raw_metric_delta = max(0.0, float(latest.primary_metric_value - prev.primary_metric_value))
    raw_likes_delta = max(0.0, float(latest.likes - prev.likes))

    if dt_hours > 0:
        metric_delta_per_hour = raw_metric_delta / dt_hours
        likes_delta_per_hour = raw_likes_delta / dt_hours
    else:
        metric_delta_per_hour = raw_metric_delta
        likes_delta_per_hour = raw_likes_delta

    # ── Rolling hourly baseline over all available history ─────────────────
    total_hours = _delta_hours(oldest.collected_at, latest.collected_at)
    total_metric_gain = max(0.0, float(latest.primary_metric_value - oldest.primary_metric_value))

    if total_hours > 0:
        baseline = total_metric_gain / total_hours
    else:
        baseline = 0.0

    # ── Baseline coverage ──────────────────────────────────────────────────
    coverage = BaselineCoverage(
        requested_hours=requested_baseline_hours,
        available_hours=total_hours,
        sufficient=total_hours >= requested_baseline_hours,
    )

    return ContentMetrics(
        current_hour_delta_views=metric_delta_per_hour,
        seven_day_rolling_hourly_baseline=baseline,
        one_hour_delta_likes=likes_delta_per_hour,
        one_hour_delta_views=metric_delta_per_hour,
        total_views=total_metric,
        total_likes=total_likes_val,
    ), coverage


def metrics_from_history(record: NormalizedContentRecord) -> ContentMetrics:
    """
    Derive ContentMetrics from real history snapshots.

    Backward-compatible wrapper around metrics_and_coverage_from_history().
    Existing callers that do not need coverage information continue to work
    without modification.
    """
    metrics, _ = metrics_and_coverage_from_history(record)
    return metrics


def _delta_hours(t1, t2) -> float:
    """Return the positive hour difference between two aware datetimes."""
    delta: timedelta = t2 - t1
    return max(0.0, delta.total_seconds() / 3600.0)

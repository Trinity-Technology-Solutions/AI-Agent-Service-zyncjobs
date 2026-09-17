"""
XGBoost readiness gating.

Evaluates whether the current data in the AI history tables meets the minimum
requirements for chronological train/validation/test splitting and meaningful
XGBoost training.  Returns WAITING_FOR_DATA when insufficient.

Criteria (all must pass):
  1. Total time span >= 48 hours
  2. >= 20 distinct content items with > 1 observation each
  3. >= 10 content items where at least one metric (primary or likes) changed
     across observations
  4. A chronological train/val/test split (60/20/20) of the qualifying
     observations must produce at least 10 samples in the test split

This module MUST NOT:
  - Call an LLM
  - Fabricate or duplicate observations
  - Change any thresholds without env config
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from typing import List, Optional

from app.domain.models import NormalizedContentRecord


@dataclass
class ReadinessReport:
    ready: bool
    status: str          # "READY" | "WAITING_FOR_DATA"
    reason: str
    details: dict = field(default_factory=dict)

    # ── Four explicit XGBoost pipeline states ──────────────────────────────
    # These must NEVER be conflated. "READY" only means data prerequisites are
    # met. It does NOT mean a model is trained, loaded, or predicting.
    #
    # data_prerequisites_met:
    #   True when the 4 readiness criteria are satisfied (time span, multi-obs
    #   items, metric variation, chronological split size).
    #
    # model_trained:
    #   True only when an actual XGBoost model artefact has been trained offline
    #   and exists on disk. Currently: prototype v0.1.0 exists as a reference
    #   implementation but is NOT part of the production pipeline.
    #
    # model_loaded:
    #   True only when a trained model is actively loaded in memory and available
    #   for inference. Currently: False — no model is loaded in production.
    #
    # prediction_available:
    #   True only when model_loaded=True AND the model is actively serving
    #   predictions for incoming content.  Currently: False.
    #   Deterministic classification (velocity ratio, like acceleration) is the
    #   sole authoritative classification mechanism.
    data_prerequisites_met: bool = False
    model_trained: bool = False        # prototype exists offline; not in production pipeline
    model_loaded: bool = False         # no model loaded in production
    prediction_available: bool = False  # deterministic gating is authoritative


# Minimum requirements (hard-coded sensible defaults; no env override needed
# because these are ML training prerequisites, not business thresholds)
_MIN_SPAN_HOURS: float = 48.0
_MIN_MULTI_OBS_ITEMS: int = 20
_MIN_CHANGING_METRIC_ITEMS: int = 10
_MIN_TEST_SAMPLES: int = 10


def check_xgboost_readiness(records: List[NormalizedContentRecord]) -> ReadinessReport:
    """
    Evaluate whether records are sufficient for XGBoost training.

    Parameters
    ----------
    records:
        All NormalizedContentRecord objects fetched for a given platform.

    Returns
    -------
    ReadinessReport with ready=True only when all four criteria pass.
    """
    if not records:
        return ReadinessReport(
            ready=False,
            status="WAITING_FOR_DATA",
            reason="No content records available.",
            details={"total_records": 0},
            data_prerequisites_met=False,
            model_trained=False,
            model_loaded=False,
            prediction_available=False,
        )

    # ── 1. Total time span ────────────────────────────────────────────────
    all_timestamps = []
    for rec in records:
        for snap in rec.history:
            ts = snap.collected_at
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            all_timestamps.append(ts)

    if not all_timestamps:
        return ReadinessReport(
            ready=False,
            status="WAITING_FOR_DATA",
            reason="No history snapshots found across any content item.",
            details={"total_records": len(records)},
            data_prerequisites_met=False,
            model_trained=False,
            model_loaded=False,
            prediction_available=False,
        )

    span_hours = (max(all_timestamps) - min(all_timestamps)).total_seconds() / 3600.0

    if span_hours < _MIN_SPAN_HOURS:
        return ReadinessReport(
            ready=False,
            status="WAITING_FOR_DATA",
            reason=(
                f"Total data span is only {span_hours:.1f}h — "
                f"need at least {_MIN_SPAN_HOURS:.0f}h for chronological splitting."
            ),
            details={
                "span_hours": round(span_hours, 2),
                "required_span_hours": _MIN_SPAN_HOURS,
                "total_content_items": len(records),
            },
            data_prerequisites_met=False,
            model_trained=False,
            model_loaded=False,
            prediction_available=False,
        )

    # ── 2. Items with multiple observations ───────────────────────────────
    multi_obs_items = [rec for rec in records if len(rec.history) > 1]

    if len(multi_obs_items) < _MIN_MULTI_OBS_ITEMS:
        return ReadinessReport(
            ready=False,
            status="WAITING_FOR_DATA",
            reason=(
                f"Only {len(multi_obs_items)} content items have >1 observation — "
                f"need at least {_MIN_MULTI_OBS_ITEMS} for meaningful feature engineering."
            ),
            details={
                "span_hours": round(span_hours, 2),
                "multi_obs_items": len(multi_obs_items),
                "required_multi_obs_items": _MIN_MULTI_OBS_ITEMS,
            },
            data_prerequisites_met=False,
            model_trained=False,
            model_loaded=False,
            prediction_available=False,
        )

    # ── 3. Items with changing metrics ────────────────────────────────────
    changing_items = 0
    for rec in multi_obs_items:
        primary_values = [s.primary_metric_value for s in rec.history]
        likes_values = [s.likes for s in rec.history]
        if max(primary_values) != min(primary_values) or max(likes_values) != min(likes_values):
            changing_items += 1

    if changing_items < _MIN_CHANGING_METRIC_ITEMS:
        return ReadinessReport(
            ready=False,
            status="WAITING_FOR_DATA",
            reason=(
                f"Only {changing_items} content items show metric variation — "
                f"need at least {_MIN_CHANGING_METRIC_ITEMS} to build meaningful features."
            ),
            details={
                "span_hours": round(span_hours, 2),
                "multi_obs_items": len(multi_obs_items),
                "changing_metric_items": changing_items,
                "required_changing_metric_items": _MIN_CHANGING_METRIC_ITEMS,
            },
            data_prerequisites_met=False,
            model_trained=False,
            model_loaded=False,
            prediction_available=False,
        )

    # ── 4. Chronological split feasibility ───────────────────────────────
    # Count total qualifying observations (only from multi-obs items)
    total_qualifying_obs = sum(len(rec.history) for rec in multi_obs_items)
    test_samples = int(total_qualifying_obs * 0.20)

    if test_samples < _MIN_TEST_SAMPLES:
        return ReadinessReport(
            ready=False,
            status="WAITING_FOR_DATA",
            reason=(
                f"Chronological 20% test split yields only {test_samples} samples — "
                f"need at least {_MIN_TEST_SAMPLES} test samples."
            ),
            details={
                "span_hours": round(span_hours, 2),
                "total_qualifying_observations": total_qualifying_obs,
                "estimated_test_samples": test_samples,
                "required_test_samples": _MIN_TEST_SAMPLES,
            },
            data_prerequisites_met=False,
            model_trained=False,
            model_loaded=False,
            prediction_available=False,
        )

    return ReadinessReport(
        ready=True,
        status="READY",
        reason=(
            "Data prerequisites met — XGBoost training pipeline is unblocked. "
            "Note: no model is currently trained, loaded, or serving predictions. "
            "Deterministic classification (velocity ratio, like acceleration) remains authoritative."
        ),
        details={
            "span_hours": round(span_hours, 2),
            "total_content_items": len(records),
            "multi_obs_items": len(multi_obs_items),
            "changing_metric_items": changing_items,
            "total_qualifying_observations": total_qualifying_obs,
            "estimated_test_samples": test_samples,
        },
        data_prerequisites_met=True,
        model_trained=False,      # prototype exists offline; not in production pipeline
        model_loaded=False,       # no model loaded in production
        prediction_available=False,  # deterministic gating is authoritative
    )


# ── In-Memory Readiness Cache ──────────────────────────────────────────
_PLATFORM_READINESS: dict[str, ReadinessReport] = {}


def set_cached_readiness(platform: str, report: ReadinessReport) -> None:
    """Store the evaluated readiness report for a platform."""
    _PLATFORM_READINESS[platform] = report


def get_cached_readiness(platform: Optional[str] = None) -> str:
    """
    Get the current XGBoost readiness status ('READY' or 'WAITING_FOR_DATA').
    If a specific platform is requested, checks that platform's report.
    If no platform is requested, returns 'READY' if any platform is ready.
    """
    if platform and platform in _PLATFORM_READINESS:
        return _PLATFORM_READINESS[platform].status
    for rep in _PLATFORM_READINESS.values():
        if rep.ready:
            return "READY"
    return "WAITING_FOR_DATA"


def get_readiness_report(platform: Optional[str] = None) -> Optional[ReadinessReport]:
    """Retrieve full ReadinessReport object for a platform if cached."""
    if platform and platform in _PLATFORM_READINESS:
        return _PLATFORM_READINESS[platform]
    for rep in _PLATFORM_READINESS.values():
        return rep
    return None


def get_xgboost_status_detail(platform: Optional[str] = None) -> dict:
    """
    Return the full 4-state XGBoost status for use in API responses.

    Always returns all four states so callers never claim more than is true.
    Never returns 'READY' as meaning 'model is predicting' — that would be false.
    """
    report = get_readiness_report(platform)
    if report is None:
        return {
            "status": "WAITING_FOR_DATA",
            "data_prerequisites_met": False,
            "model_trained": False,
            "model_loaded": False,
            "prediction_available": False,
            "reason": "Readiness not yet evaluated for this platform.",
            "note": (
                "Deterministic classification (velocity ratio, like acceleration) "
                "is the sole authoritative classification mechanism. "
                "XGBoost predictions are NOT currently active."
            ),
        }
    return {
        "status": report.status,
        "data_prerequisites_met": report.data_prerequisites_met,
        "model_trained": report.model_trained,
        "model_loaded": report.model_loaded,
        "prediction_available": report.prediction_available,
        "reason": report.reason,
        "details": report.details,
        "note": (
            "Deterministic classification (velocity ratio, like acceleration) "
            "is the sole authoritative classification mechanism. "
            "XGBoost predictions are NOT currently active."
        ),
    }


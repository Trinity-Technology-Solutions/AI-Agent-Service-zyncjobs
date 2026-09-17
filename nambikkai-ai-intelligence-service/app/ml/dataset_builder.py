"""
Track A — XGBoost Numerical Intelligence Training Dataset Foundation.

Provides leakage-safe, point-in-time feature extraction, target construction,
and embargoed temporal dataset splitting for machine learning training datasets.

Leakage prevention contract:
  For any prediction cutoff timestamp T0:
    Snapshots <= T0   -->  Features(T0)
    Snapshots > T0    -->  Target(T0, Horizon)

This module MUST NOT:
  - Modify production /analyze routes or gating logic.
  - Hardcode credentials, thresholds, or DB settings.
  - Use snapshots > T0 in feature engineering.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.domain.models import ContentMetadata, HistorySnapshot, NormalizedContentRecord


class MLDatasetRow(BaseModel):
    """Represents a single leakage-safe point-in-time ML training example."""

    content_id: str
    platform: str
    prediction_timestamp: datetime
    features: dict[str, float]
    target_surge: int = Field(ge=0, le=1)
    future_velocity_ratio: float


class DatasetReadinessReport(BaseModel):
    """Summary of ML dataset readiness and split characteristics."""

    total_rows: int
    platform_counts: dict[str, int]
    positive_labels: int
    negative_labels: int
    excluded_unlabeled_rows: int
    min_history_coverage_hours: float
    max_history_coverage_hours: float
    unique_content_count: int
    split_counts: dict[str, int]


def extract_point_in_time_features(
    history: list[HistorySnapshot],
    cutoff_t0: datetime,
    metadata: Optional[ContentMetadata] = None,
    platform: str = "youtube",
) -> dict[str, float]:
    """
    Extract numerical features strictly using history snapshots where collected_at <= cutoff_t0.

    Raises:
        ValueError: If fewer than 2 snapshots exist prior to or at cutoff_t0.
    """
    history_t0 = [s for s in history if s.collected_at <= cutoff_t0]
    history_t0.sort(key=lambda s: s.collected_at)

    if len(history_t0) < 2:
        raise ValueError(
            f"Insufficient history at T0={cutoff_t0.isoformat()}: "
            f"found {len(history_t0)} snapshots (minimum 2 required)."
        )

    s_k = history_t0[-1]
    s_1 = history_t0[0]
    s_prev = history_t0[-2]

    t_k = s_k.collected_at
    t_1 = s_1.collected_at
    t_prev = s_prev.collected_at

    available_history_hours = max(0.0, (t_k - t_1).total_seconds() / 3600.0)
    current_metric_value = float(s_k.primary_metric_value)

    # ── Historical baseline velocity over available history <= T0 ─────────────
    if available_history_hours > 0:
        metric_gain = max(0.0, float(s_k.primary_metric_value - s_1.primary_metric_value))
        historical_baseline_velocity = metric_gain / available_history_hours
    else:
        historical_baseline_velocity = 0.0

    # ── Recent hourly metric velocity ─────────────────────────────────────────
    dt_recent = max(0.0, (t_k - t_prev).total_seconds() / 3600.0)
    delta_m_recent = max(0.0, float(s_k.primary_metric_value - s_prev.primary_metric_value))

    if dt_recent > 0:
        recent_metric_velocity = delta_m_recent / dt_recent
    else:
        recent_metric_velocity = delta_m_recent

    # ── Velocity ratio ────────────────────────────────────────────────────────
    if historical_baseline_velocity > 0:
        velocity_ratio = recent_metric_velocity / historical_baseline_velocity
    else:
        velocity_ratio = 0.0

    # ── Like engagement rate ──────────────────────────────────────────────────
    delta_likes = max(0.0, float(s_k.likes - s_prev.likes))
    like_engagement_rate = (delta_likes / max(delta_m_recent, 1.0)) * 100.0

    # ── Metric acceleration (rate of change of velocity) ─────────────────────
    if len(history_t0) >= 3:
        s_prev2 = history_t0[-3]
        dt_prev = max(0.0, (t_prev - s_prev2.collected_at).total_seconds() / 3600.0)
        delta_m_prev = max(0.0, float(s_prev.primary_metric_value - s_prev2.primary_metric_value))
        prev_metric_velocity = delta_m_prev / dt_prev if dt_prev > 0 else delta_m_prev

        if dt_recent > 0:
            metric_acceleration = (recent_metric_velocity - prev_metric_velocity) / dt_recent
        else:
            metric_acceleration = 0.0
    else:
        metric_acceleration = 0.0

    # ── Content age ───────────────────────────────────────────────────────────
    pub_at = s_k.published_at
    if pub_at is None and metadata is not None and metadata.published_at is not None:
        pub_at = metadata.published_at

    if pub_at is not None:
        content_age_hours = max(0.0, (t_k - pub_at).total_seconds() / 3600.0)
    else:
        content_age_hours = 0.0

    return {
        "current_metric_value": current_metric_value,
        "recent_metric_velocity": recent_metric_velocity,
        "historical_baseline_velocity": historical_baseline_velocity,
        "velocity_ratio": velocity_ratio,
        "like_engagement_rate": like_engagement_rate,
        "metric_acceleration": metric_acceleration,
        "total_likes": float(s_k.likes),
        "total_comments": float(s_k.comments),
        "content_age_hours": content_age_hours,
        "available_history_hours": available_history_hours,
        "snapshot_count": float(len(history_t0)),
        "is_baseline_7day_complete": 1.0 if available_history_hours >= 168.0 else 0.0,
    }


def compute_target_label(
    history: list[HistorySnapshot],
    cutoff_t0: datetime,
    baseline_velocity_at_t0: float,
    horizon_hours: float = 6.0,
    surge_threshold: float = 3.0,
) -> tuple[Optional[int], Optional[float]]:
    """
    Compute deterministic target surge label Y in {0, 1} strictly using future snapshots.

    Target observation window: (cutoff_t0, cutoff_t0 + horizon_hours]

    Returns:
        (target_surge, future_velocity_ratio) or (None, None) if no future snapshot exists.
    """
    future_snapshots = [
        s for s in history
        if cutoff_t0 < s.collected_at <= cutoff_t0 + timedelta(hours=horizon_hours)
    ]
    if not future_snapshots:
        return None, None

    future_snapshots.sort(key=lambda s: s.collected_at)
    f_target = future_snapshots[-1]

    # Retrieve snapshot at or immediately preceding T0
    history_t0 = [s for s in history if s.collected_at <= cutoff_t0]
    if not history_t0:
        return None, None
    s_k = max(history_t0, key=lambda s: s.collected_at)

    dt_future = max(0.0, (f_target.collected_at - s_k.collected_at).total_seconds() / 3600.0)
    delta_future_metric = max(0.0, float(f_target.primary_metric_value - s_k.primary_metric_value))

    if dt_future > 0:
        future_velocity = delta_future_metric / dt_future
    else:
        future_velocity = delta_future_metric

    if baseline_velocity_at_t0 > 0:
        future_velocity_ratio = future_velocity / baseline_velocity_at_t0
    else:
        future_velocity_ratio = 0.0

    target_surge = 1 if future_velocity_ratio >= surge_threshold else 0
    return target_surge, future_velocity_ratio


def generate_dataset_rows(
    record: NormalizedContentRecord,
    horizon_hours: float = 6.0,
    surge_threshold: float = 3.0,
) -> tuple[list[MLDatasetRow], int]:
    """
    Generate point-in-time training rows for a single content item.

    Returns:
        (valid_rows, excluded_count)
    """
    snapshots = sorted(record.history, key=lambda s: s.collected_at)
    if len(snapshots) < 2:
        return [], 0

    metadata = ContentMetadata(
        content_id=record.content_id,
        title=record.title or record.caption or f"{record.platform}/{record.content_id}",
        creator_id=record.creator_id or record.account_key or "unknown",
        platform=record.platform,
        published_at=record.content_published_at,
    )

    rows: list[MLDatasetRow] = []
    excluded_count = 0

    # Every snapshot from the 2nd onwards can serve as a candidate prediction cutoff T0
    for i in range(1, len(snapshots)):
        cutoff_t0 = snapshots[i].collected_at

        try:
            features = extract_point_in_time_features(
                history=snapshots,
                cutoff_t0=cutoff_t0,
                metadata=metadata,
                platform=record.platform,
            )
        except ValueError:
            excluded_count += 1
            continue

        baseline_velocity = features["historical_baseline_velocity"]
        target_surge, future_ratio = compute_target_label(
            history=snapshots,
            cutoff_t0=cutoff_t0,
            baseline_velocity_at_t0=baseline_velocity,
            horizon_hours=horizon_hours,
            surge_threshold=surge_threshold,
        )

        if target_surge is None or future_ratio is None:
            excluded_count += 1
            continue

        rows.append(
            MLDatasetRow(
                content_id=record.content_id,
                platform=record.platform,
                prediction_timestamp=cutoff_t0,
                features=features,
                target_surge=target_surge,
                future_velocity_ratio=future_ratio,
            )
        )

    return rows, excluded_count


def create_embargoed_temporal_split(
    rows: list[MLDatasetRow],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    horizon_hours: float = 6.0,
) -> dict[str, list[MLDatasetRow]]:
    """
    Chronologically split rows into train, validation, and test sets with a temporal embargo gap.

    Embargo window length = horizon_hours. Rows falling within embargo windows are excluded
    from training and validation to guarantee zero target horizon overlap into subsequent features.
    """
    if not rows:
        return {"train": [], "val": [], "test": [], "embargoed": []}

    sorted_rows = sorted(rows, key=lambda r: r.prediction_timestamp)
    t_min = sorted_rows[0].prediction_timestamp
    t_max = sorted_rows[-1].prediction_timestamp
    total_seconds = (t_max - t_min).total_seconds()

    if total_seconds <= 0:
        return {"train": sorted_rows, "val": [], "test": [], "embargoed": []}

    t_train_end = t_min + timedelta(seconds=total_seconds * train_ratio)
    t_val_end = t_min + timedelta(seconds=total_seconds * (train_ratio + val_ratio))
    embargo_delta = timedelta(hours=horizon_hours)

    train_set: list[MLDatasetRow] = []
    val_set: list[MLDatasetRow] = []
    test_set: list[MLDatasetRow] = []
    embargoed_set: list[MLDatasetRow] = []

    for r in sorted_rows:
        ts = r.prediction_timestamp
        if ts <= t_train_end:
            train_set.append(r)
        elif t_train_end < ts <= t_train_end + embargo_delta:
            embargoed_set.append(r)
        elif t_train_end + embargo_delta < ts <= t_val_end:
            val_set.append(r)
        elif t_val_end < ts <= t_val_end + embargo_delta:
            embargoed_set.append(r)
        else:
            test_set.append(r)

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "embargoed": embargoed_set,
    }


def generate_full_dataset(
    records: list[NormalizedContentRecord],
    horizon_hours: float = 6.0,
    surge_threshold: float = 3.0,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[list[MLDatasetRow], dict[str, list[MLDatasetRow]], DatasetReadinessReport]:
    """
    Build full training dataset and generate readiness statistics.

    Returns:
        (all_rows, split_dict, report)
    """
    all_rows: list[MLDatasetRow] = []
    total_excluded = 0
    platform_counts: dict[str, int] = {}
    unique_contents = set()

    for rec in records:
        rows, excl = generate_dataset_rows(
            record=rec,
            horizon_hours=horizon_hours,
            surge_threshold=surge_threshold,
        )
        all_rows.extend(rows)
        total_excluded += excl
        if rows:
            unique_contents.add(rec.content_id)
            platform_counts[rec.platform] = platform_counts.get(rec.platform, 0) + len(rows)

    splits = create_embargoed_temporal_split(
        rows=all_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        horizon_hours=horizon_hours,
    )

    pos_count = sum(1 for r in all_rows if r.target_surge == 1)
    neg_count = sum(1 for r in all_rows if r.target_surge == 0)

    if all_rows:
        coverages = [r.features["available_history_hours"] for r in all_rows]
        min_cov = min(coverages)
        max_cov = max(coverages)
    else:
        min_cov = 0.0
        max_cov = 0.0

    report = DatasetReadinessReport(
        total_rows=len(all_rows),
        platform_counts=platform_counts,
        positive_labels=pos_count,
        negative_labels=neg_count,
        excluded_unlabeled_rows=total_excluded,
        min_history_coverage_hours=min_cov,
        max_history_coverage_hours=max_cov,
        unique_content_count=len(unique_contents),
        split_counts={k: len(v) for k, v in splits.items()},
    )

    return all_rows, splits, report

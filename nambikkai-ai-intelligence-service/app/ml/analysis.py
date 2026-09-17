"""
Track A — XGBoost Offline Model Analysis & Threshold Evaluation.

Provides offline explainability routines for feature importance, prediction
probability distributions, classification threshold trade-offs, and reference
baseline comparison.

Must NOT be connected to live production /analyze inference in this task.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import xgboost as xgb

from app.domain.models import NormalizedContentRecord
from app.ml.dataset_builder import MLDatasetRow, generate_full_dataset
from app.ml.trainer import (
    FEATURE_NAMES,
    evaluate_baseline_rule,
    evaluate_predictions,
    load_model_artifact,
    prepare_feature_matrix,
)


def analyze_feature_importance(model: xgb.XGBClassifier) -> list[dict[str, Any]]:
    """
    Compute model-native feature importance scores mapped to FEATURE_NAMES.

    Returns all 12 features sorted deterministically by importance descending
    (with feature_name as tie-breaker).
    """
    booster = model.get_booster()
    # Try gain importance from booster, falling back to feature_importances_
    score_dict = booster.get_score(importance_type="gain")

    raw_scores: dict[str, float] = {}

    if score_dict:
        # Score keys can be feature names ("velocity_ratio") or position strings ("f0", "f1")
        for k, val in score_dict.items():
            if k in FEATURE_NAMES:
                raw_scores[k] = float(val)
            elif k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(FEATURE_NAMES):
                    raw_scores[FEATURE_NAMES[idx]] = float(val)
    elif hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        for idx, val in enumerate(model.feature_importances_):
            if idx < len(FEATURE_NAMES):
                raw_scores[FEATURE_NAMES[idx]] = float(val)

    items: list[dict[str, Any]] = []
    for fname in FEATURE_NAMES:
        importance = raw_scores.get(fname, 0.0)
        items.append({"feature_name": fname, "importance_score": float(importance)})

    # Sort by importance descending, then feature_name ascending for deterministic ranking
    items.sort(key=lambda x: (-x["importance_score"], x["feature_name"]))

    for rank, item in enumerate(items, start=1):
        item["rank"] = rank

    return items


def _compute_stats(arr: np.ndarray) -> dict[str, Optional[float]]:
    if len(arr) == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
        }

    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def analyze_prediction_distribution(
    rows: list[MLDatasetRow],
    model: xgb.XGBClassifier,
) -> dict[str, Any]:
    """
    Analyze probability distribution P(Y=1) for actual positive vs negative rows.
    """
    if not rows:
        return {
            "total_rows": 0,
            "positive_class": _compute_stats(np.array([])),
            "negative_class": _compute_stats(np.array([])),
        }

    X, y = prepare_feature_matrix(rows)
    probs = model.predict_proba(X)[:, 1]

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    pos_probs = probs[pos_mask]
    neg_probs = probs[neg_mask]

    return {
        "total_rows": len(rows),
        "positive_class": _compute_stats(pos_probs),
        "negative_class": _compute_stats(neg_probs),
    }


def analyze_thresholds(
    rows: list[MLDatasetRow],
    model: xgb.XGBClassifier,
    thresholds: Optional[list[float]] = None,
) -> list[dict[str, Any]]:
    """
    Evaluate model classification metrics across candidate probability thresholds.
    """
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    for t in thresholds:
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"Threshold {t} out of range [0.0, 1.0]")

    if not rows:
        results = []
        for t in thresholds:
            m = evaluate_predictions(np.array([], dtype=np.int32), np.array([], dtype=np.float32), threshold=t)
            m["threshold"] = t
            results.append(m)
        return results

    X, y = prepare_feature_matrix(rows)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for t in thresholds:
        m = evaluate_predictions(y, probs, threshold=t)
        m["threshold"] = t
        results.append(m)

    return results


def run_full_model_analysis(
    artifact_dir: Union[str, Path],
    records: list[NormalizedContentRecord],
    horizon_hours: float = 6.0,
    surge_threshold: float = 3.0,
) -> dict[str, Any]:
    """
    Load existing model artifact and run full feature importance, probability,
    and threshold analysis against records dataset.

    Raises:
        FileNotFoundError: If the model artifact directory does not contain valid files.
    """
    inp_dir = Path(artifact_dir)
    model, metadata = load_model_artifact(inp_dir)

    all_rows, splits, dataset_report = generate_full_dataset(
        records=records,
        horizon_hours=horizon_hours,
        surge_threshold=surge_threshold,
    )

    importance = analyze_feature_importance(model)
    dist = analyze_prediction_distribution(all_rows, model)
    threshold_eval = analyze_thresholds(all_rows, model)

    baseline_eval = evaluate_baseline_rule(all_rows, surge_threshold)

    return {
        "analysis_metadata": {
            "model_version": metadata.get("model_version"),
            "training_timestamp": metadata.get("training_timestamp"),
            "total_rows_analyzed": dataset_report.total_rows,
            "unique_content_count": dataset_report.unique_content_count,
            "positive_labels": dataset_report.positive_labels,
            "negative_labels": dataset_report.negative_labels,
        },
        "feature_importance": importance,
        "probability_distribution": dist,
        "threshold_analysis": threshold_eval,
        "reference_baseline": baseline_eval,
    }

"""
Track A — XGBoost Numerical Intelligence Model Training Pipeline.

Provides offline training, imbalanced binary classification evaluation,
deterministic baseline comparison, and artifact persistence.

Must NOT be connected to live production /analyze inference in this task.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union, cast

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.ml.dataset_builder import MLDatasetRow

# ── Centralized Authoritative Feature Contract ──────────────────────────────
FEATURE_NAMES: list[str] = [
    "current_metric_value",
    "recent_metric_velocity",
    "historical_baseline_velocity",
    "velocity_ratio",
    "like_engagement_rate",
    "metric_acceleration",
    "total_likes",
    "total_comments",
    "content_age_hours",
    "available_history_hours",
    "snapshot_count",
    "is_baseline_7day_complete",
]


def prepare_feature_matrix(rows: list[MLDatasetRow]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 2D NumPy feature matrix X and 1D target vector y from dataset rows.

    Columns strictly follow FEATURE_NAMES in exact order.
    Excludes content_id, title, caption, timestamps, and future data.
    """
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X_list: list[list[float]] = []
    y_list: list[int] = []

    for r in rows:
        row_feats: list[float] = []
        for fname in FEATURE_NAMES:
            if fname not in r.features:
                raise ValueError(
                    f"Feature '{fname}' missing from dataset row for content_id='{r.content_id}' "
                    f"at {r.prediction_timestamp.isoformat()}"
                )
            row_feats.append(float(r.features[fname]))

        X_list.append(row_feats)
        y_list.append(r.target_surge)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute binary classification evaluation metrics.

    Handles empty splits or single-class distributions gracefully by setting
    metric values to None internally rather than failing.
    """
    sample_count = len(y_true)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": 0.0,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "pr_auc": None,
            "roc_auc": None,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "status": "unavailable_empty_split",
        }

    pos_count = int(np.sum(y_true == 1))
    neg_count = int(np.sum(y_true == 0))
    pos_rate = float(pos_count / sample_count)

    y_pred = (y_pred_prob >= threshold).astype(np.int32)

    # Confusion matrix elements
    if pos_count > 0 and neg_count > 0:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Metrics calculation with zero-division protection
    prec: Optional[float] = float(precision_score(y_true, y_pred, zero_division=cast(Any, 0))) if (tp + fp) > 0 else None
    rec: Optional[float] = float(recall_score(y_true, y_pred, zero_division=cast(Any, 0))) if pos_count > 0 else None
    f1: Optional[float] = float(f1_score(y_true, y_pred, zero_division=cast(Any, 0))) if (pos_count > 0 and (tp + fp) > 0) else None

    # AUC calculation requires presence of positive class
    pr_auc: Optional[float] = float(average_precision_score(y_true, y_pred_prob)) if pos_count > 0 else None
    roc_auc: Optional[float] = (
        float(roc_auc_score(y_true, y_pred_prob)) if (pos_count > 0 and neg_count > 0) else None
    )

    return {
        "sample_count": sample_count,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_rate": pos_rate,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "status": "evaluated",
    }


def evaluate_baseline_rule(
    rows: list[MLDatasetRow],
    velocity_threshold: float = 3.0,
) -> dict[str, Any]:
    """
    Evaluate deterministic reference heuristic: predict 1 if velocity_ratio >= threshold.
    """
    if not rows:
        return evaluate_predictions(np.array([], dtype=np.int32), np.array([], dtype=np.float32))

    y_true = np.array([r.target_surge for r in rows], dtype=np.int32)
    # Predict 1.0 probability if velocity_ratio >= threshold, else 0.0
    y_pred_prob = np.array(
        [1.0 if r.features.get("velocity_ratio", 0.0) >= velocity_threshold else 0.0 for r in rows],
        dtype=np.float32,
    )
    res = evaluate_predictions(y_true, y_pred_prob, threshold=0.5)
    res["baseline_rule"] = f"velocity_ratio >= {velocity_threshold}"
    return res


def save_model_artifact(
    model: xgb.XGBClassifier,
    metadata: dict[str, Any],
    artifact_dir: Union[str, Path],
) -> tuple[Path, Path]:
    """
    Save trained XGBoost model (native JSON) and metadata JSON to artifact directory.
    """
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgboost_model.json"
    metadata_path = out_dir / "metadata.json"

    model.save_model(str(model_path))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return model_path, metadata_path


def load_model_artifact(artifact_dir: Union[str, Path]) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    """
    Load trained XGBoost model and metadata from artifact directory.
    """
    inp_dir = Path(artifact_dir)
    model_path = inp_dir / "xgboost_model.json"
    metadata_path = inp_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return model, metadata


def train_track_a_model(
    all_rows: list[MLDatasetRow],
    splits: dict[str, list[MLDatasetRow]],
    config: Optional[dict[str, Any]] = None,
    artifact_dir: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """
    Train and evaluate Track A XGBoost binary classification model.

    Raises:
        ValueError: If training split has zero positive or zero negative examples.
    """
    if config is None:
        config = {}

    default_config = {
        "n_estimators": 50,
        "max_depth": 3,
        "learning_rate": 0.05,
        "random_state": 42,
        "horizon_hours": 6.0,
        "surge_threshold": 3.0,
        "model_version": "0.1.0-prototype",
    }
    merged_config = {**default_config, **config}

    train_rows = splits.get("train", [])
    val_rows = splits.get("val", [])
    test_rows = splits.get("test", [])

    X_train, y_train = prepare_feature_matrix(train_rows)

    pos_train_count = int(np.sum(y_train == 1))
    neg_train_count = int(np.sum(y_train == 0))

    if pos_train_count == 0:
        raise ValueError(
            f"Cannot train model: training split has 0 positive examples "
            f"(found {neg_train_count} negative examples)."
        )
    if neg_train_count == 0:
        raise ValueError("Cannot train model: training split has 0 negative examples.")

    scale_pos_weight = float(neg_train_count / pos_train_count)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=merged_config["n_estimators"],
        max_depth=merged_config["max_depth"],
        learning_rate=merged_config["learning_rate"],
        scale_pos_weight=scale_pos_weight,
        random_state=merged_config["random_state"],
        eval_metric="logloss",
    )

    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_train_prob = model.predict_proba(X_train)[:, 1] if len(X_train) > 0 else np.array([], dtype=np.float32)
    train_metrics = evaluate_predictions(y_train, y_train_prob)

    X_val, y_val = prepare_feature_matrix(val_rows)
    if len(X_val) > 0:
        y_val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = evaluate_predictions(y_val, y_val_prob)
    else:
        val_metrics = evaluate_predictions(np.array([], dtype=np.int32), np.array([], dtype=np.float32))

    X_test, y_test = prepare_feature_matrix(test_rows)
    if len(X_test) > 0:
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_predictions(y_test, y_test_prob)
    else:
        test_metrics = evaluate_predictions(np.array([], dtype=np.int32), np.array([], dtype=np.float32))

    # ── Baseline evaluation ───────────────────────────────────────────────────
    surge_threshold = float(merged_config["surge_threshold"])
    train_baseline = evaluate_baseline_rule(train_rows, surge_threshold)
    val_baseline = evaluate_baseline_rule(val_rows, surge_threshold)
    test_baseline = evaluate_baseline_rule(test_rows, surge_threshold)

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata = {
        "feature_names": FEATURE_NAMES,
        "model_version": merged_config["model_version"],
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction_horizon_hours": merged_config["horizon_hours"],
        "target_surge_threshold": merged_config["surge_threshold"],
        "train_row_count": len(train_rows),
        "positive_count": pos_train_count,
        "negative_count": neg_train_count,
        "scale_pos_weight": scale_pos_weight,
        "xgboost_config": merged_config,
        "training_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": {
            "train": train_baseline,
            "val": val_baseline,
            "test": test_baseline,
        },
    }

    if artifact_dir is None:
        artifact_dir = Path("models/track_a_xgboost")

    model_path, metadata_path = save_model_artifact(model, metadata, artifact_dir)

    return {
        "model": model,
        "metadata": metadata,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }

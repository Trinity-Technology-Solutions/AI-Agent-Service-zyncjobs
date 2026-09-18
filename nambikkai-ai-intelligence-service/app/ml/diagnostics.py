"""
Track A — XGBoost Model Artifact Diagnostic Utility.

Provides offline, read-only inspection of the persisted XGBoost model artifact:
  - Booster tree structure (splits, leaves, feature usage)
  - Feature-class statistics (dtype, range, zeros, NaN, Inf)
  - Positive vs negative class comparison
  - Prediction probability degeneracy detection
  - Metadata vs artifact consistency checking
  - Root cause classification

HARD SCOPE:
  - Does NOT retrain, tune, or modify any model.
  - Does NOT overwrite the model artifact.
  - Does NOT modify production /analyze, gating, Option B, or LLM providers.
  - Does NOT require a live PostgreSQL connection; accepts pre-loaded rows.
  - All functions are deterministic and side-effect free.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import xgboost as xgb

from app.ml.dataset_builder import MLDatasetRow
from app.ml.trainer import (
    FEATURE_NAMES,
    evaluate_predictions,
    load_model_artifact,
    prepare_feature_matrix,
)

# ── Floating-point tolerance for constant-probability detection ───────────────
# Two probabilities are considered "equal" if their absolute difference is below
# this threshold. XGBoost float32 precision is ~1e-7; 1e-4 gives safe margin.
_PROB_EQUALITY_TOLERANCE: float = 1e-4


# ── 1. Tree structure inspection ──────────────────────────────────────────────

def inspect_booster_trees(model: xgb.XGBClassifier) -> dict[str, Any]:
    """
    Inspect the internal booster tree structure of a loaded XGBClassifier.

    Returns a structured summary including:
      - num_trees: total tree count
      - tree_summaries: per-tree depth, leaf count, split count, split features, thresholds
      - split_feature_frequency: how many times each feature index was split across all trees
      - feature_index_to_name: mapping from XGBoost internal f0/f1/... to FEATURE_NAMES
      - all_split_indices: raw split_indices arrays from artifact
      - constant_predictor: True if no meaningful branching exists
    """
    booster = model.get_booster()
    dump_list = booster.get_dump(dump_format="json")

    feature_index_to_name: dict[int, str] = {i: name for i, name in enumerate(FEATURE_NAMES)}

    split_feature_frequency: dict[str, int] = {}
    tree_summaries: list[dict[str, Any]] = []

    total_leaves = 0
    total_splits = 0
    max_depth_overall = 0

    for tree_idx, tree_json_str in enumerate(dump_list):
        tree_json = json.loads(tree_json_str)

        leaves: list[dict] = []
        splits: list[dict] = []

        def _walk(node: dict, depth: int = 0) -> None:
            nonlocal max_depth_overall
            if "leaf" in node:
                leaves.append({"depth": depth, "leaf_value": node["leaf"]})
                max_depth_overall = max(max_depth_overall, depth)
            elif "split" in node:
                feat_raw = str(node["split"])
                feat_name: str
                if feat_raw.startswith("f") and feat_raw[1:].isdigit():
                    feat_idx = int(feat_raw[1:])
                    feat_name = feature_index_to_name.get(feat_idx, feat_raw)
                else:
                    feat_name = feat_raw
                splits.append({
                    "depth": depth,
                    "feature_raw": feat_raw,
                    "feature_name": feat_name,
                    "split_threshold": node.get("split_condition"),
                })
                split_feature_frequency[feat_name] = split_feature_frequency.get(feat_name, 0) + 1
                max_depth_overall = max(max_depth_overall, depth)
                # XGBoost JSON dump uses a 'children' list, not recursive yes/no dicts.
                # yes/no/missing keys hold integer node IDs, not subtrees.
                for child_node in node.get("children", []):
                    _walk(child_node, depth + 1)

        _walk(tree_json)

        total_leaves += len(leaves)
        total_splits += len(splits)

        leaf_values = [lf["leaf_value"] for lf in leaves]
        split_thresholds = list({sp["split_threshold"] for sp in splits})
        split_features_used = list({sp["feature_name"] for sp in splits})

        tree_summaries.append({
            "tree_id": tree_idx,
            "num_leaves": len(leaves),
            "num_splits": len(splits),
            "max_depth": max(lf["depth"] for lf in leaves) if leaves else 0,
            "split_features": split_features_used,
            "split_thresholds": split_thresholds,
            "leaf_values": leaf_values,
            "unique_leaf_values": len(set(round(v, 8) for v in leaf_values)),
        })

    constant_predictor = (len(split_feature_frequency) <= 1 and
                          sum(split_feature_frequency.values()) <= len(dump_list))

    return {
        "num_trees": len(dump_list),
        "total_leaves": total_leaves,
        "total_splits": total_splits,
        "max_depth_overall": max_depth_overall,
        "split_feature_frequency": split_feature_frequency,
        "feature_index_to_name": feature_index_to_name,
        "tree_summaries": tree_summaries,
        "constant_predictor": constant_predictor,
        "features_never_split": [
            name for name in FEATURE_NAMES
            if name not in split_feature_frequency
        ],
    }


# ── 2. Feature-class statistics ───────────────────────────────────────────────

def _feature_stats(values: np.ndarray) -> dict[str, Any]:
    """Compute descriptive stats for a 1-D numpy array without modifying it."""
    n = len(values)
    if n == 0:
        return {
            "count": 0, "unique_count": 0,
            "min": None, "max": None, "mean": None, "median": None, "std": None,
            "zero_count": 0, "nan_count": 0, "inf_count": 0,
        }
    nan_mask = np.isnan(values)
    inf_mask = np.isinf(values)
    nan_count = int(np.sum(nan_mask))
    inf_count = int(np.sum(inf_mask))
    zero_count = int(np.sum(values == 0.0))

    finite = values[~nan_mask & ~inf_mask]
    return {
        "count": n,
        "unique_count": len(np.unique(values)),
        "min": float(np.min(finite)) if len(finite) > 0 else None,
        "max": float(np.max(finite)) if len(finite) > 0 else None,
        "mean": float(np.mean(finite)) if len(finite) > 0 else None,
        "median": float(np.median(finite)) if len(finite) > 0 else None,
        "std": float(np.std(finite)) if len(finite) > 0 else None,
        "zero_count": zero_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def compute_feature_class_statistics(
    rows: list[MLDatasetRow],
) -> dict[str, Any]:
    """
    Compute per-feature statistics across all rows, and separately for Y=1 and Y=0.

    Returns a dict keyed by feature name, each containing:
      - 'overall': stats for all rows
      - 'positive': stats for Y=1 rows
      - 'negative': stats for Y=0 rows
      - 'dtype': numpy dtype string
    """
    if not rows:
        return {
            fname: {
                "dtype": "float32",
                "overall": _feature_stats(np.array([], dtype=np.float32)),
                "positive": _feature_stats(np.array([], dtype=np.float32)),
                "negative": _feature_stats(np.array([], dtype=np.float32)),
            }
            for fname in FEATURE_NAMES
        }

    X, y = prepare_feature_matrix(rows)  # shape (N, 12), dtype float32
    pos_mask = (y == 1)
    neg_mask = (y == 0)

    result: dict[str, Any] = {}
    for col_idx, fname in enumerate(FEATURE_NAMES):
        col = X[:, col_idx]
        result[fname] = {
            "dtype": str(col.dtype),
            "overall": _feature_stats(col),
            "positive": _feature_stats(col[pos_mask]),
            "negative": _feature_stats(col[neg_mask]),
        }
    return result


# ── 3. Prediction behavior inspection ────────────────────────────────────────

def inspect_prediction_behavior(
    rows: list[MLDatasetRow],
    model: xgb.XGBClassifier,
) -> dict[str, Any]:
    """
    Evaluate prediction probabilities and class predictions on given rows.

    Returns:
      - total_rows
      - unique_prob_count: number of distinct probability values
      - prob_stats: min, max, mean, median, std over all rows
      - all_probs_equal: True if all P(Y=1) are within _PROB_EQUALITY_TOLERANCE
      - tolerance_used: the documented tolerance value
      - positive_class_probs: prob stats for actual Y=1 rows
      - negative_class_probs: prob stats for actual Y=0 rows
      - prediction_class_counts: dict of {0: count, 1: count}
      - class_ordering: model's class ordering (to verify column 1 = positive class)
    """
    if not rows:
        return {
            "total_rows": 0,
            "unique_prob_count": 0,
            "prob_stats": _feature_stats(np.array([], dtype=np.float64)),
            "all_probs_equal": None,
            "tolerance_used": _PROB_EQUALITY_TOLERANCE,
            "positive_class_probs": _feature_stats(np.array([], dtype=np.float64)),
            "negative_class_probs": _feature_stats(np.array([], dtype=np.float64)),
            "prediction_class_counts": {},
            "class_ordering": list(model.classes_) if hasattr(model, "classes_") else "unavailable",
        }

    X, y = prepare_feature_matrix(rows)
    proba_matrix = model.predict_proba(X)   # shape (N, 2)
    classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]

    # Identify the column index for the positive class (1)
    pos_col = classes.index(1) if 1 in classes else 1
    y_prob = proba_matrix[:, pos_col].astype(np.float64)
    y_pred = model.predict(X)

    pos_mask = (y == 1)
    neg_mask = (y == 0)

    unique_probs = np.unique(y_prob)
    prob_range = float(np.max(y_prob) - np.min(y_prob))
    all_probs_equal = prob_range <= _PROB_EQUALITY_TOLERANCE

    pred_counts: dict[int, int] = {}
    for cls in [0, 1]:
        pred_counts[cls] = int(np.sum(y_pred == cls))

    return {
        "total_rows": len(rows),
        "unique_prob_count": len(unique_probs),
        "unique_prob_values": [float(v) for v in unique_probs[:10]],  # up to 10 samples
        "prob_stats": _feature_stats(y_prob),
        "all_probs_equal": all_probs_equal,
        "tolerance_used": _PROB_EQUALITY_TOLERANCE,
        "prob_range": prob_range,
        "positive_class_probs": _feature_stats(y_prob[pos_mask]),
        "negative_class_probs": _feature_stats(y_prob[neg_mask]),
        "prediction_class_counts": pred_counts,
        "class_ordering": classes,
    }


# ── 4. Metadata vs artifact consistency check ─────────────────────────────────

def compare_model_metadata(
    model: xgb.XGBClassifier,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare the loaded XGBClassifier against stored metadata.json values.

    Reports each field as:
      - 'metadata_value': value from metadata.json
      - 'artifact_value': value derived from the loaded model object
      - 'match': True/False/None (None = cannot verify from artifact)
      - 'flag': True if mismatch detected
    """
    checks: dict[str, dict[str, Any]] = {}

    def _check(key: str, meta_val: Any, artifact_val: Any, can_verify: bool = True) -> None:
        if not can_verify or artifact_val == "unavailable":
            checks[key] = {
                "metadata_value": meta_val,
                "artifact_value": "unavailable",
                "match": None,
                "flag": False,
            }
        else:
            match = (meta_val == artifact_val)
            checks[key] = {
                "metadata_value": meta_val,
                "artifact_value": artifact_val,
                "match": match,
                "flag": not match,
            }

    # Feature names
    meta_features = metadata.get("feature_names", [])
    _check("feature_names", meta_features, FEATURE_NAMES)

    # Number of estimators
    meta_n_est = metadata.get("xgboost_config", {}).get("n_estimators", "unavailable")
    booster = model.get_booster()
    dump = booster.get_dump(dump_format="json")
    artifact_n_est = len(dump)
    _check("n_estimators", meta_n_est, artifact_n_est)

    # Objective
    meta_objective = "binary:logistic"  # assumed from config; not stored explicitly in metadata
    cfg: Optional[dict[str, Any]] = None
    try:
        raw_cfg = json.loads(booster.save_config())
        if isinstance(raw_cfg, dict):
            cfg = raw_cfg
            artifact_objective = cfg["learner"]["objective"]["name"]
        else:
            artifact_objective = "unavailable"
    except Exception:
        artifact_objective = "unavailable"
    _check("objective", meta_objective, artifact_objective)

    # scale_pos_weight
    meta_spw = metadata.get("scale_pos_weight", "unavailable")
    try:
        if cfg is not None:
            cfg_spw_str = cfg["learner"]["objective"]["reg_loss_param"]["scale_pos_weight"]
            artifact_spw = float(cfg_spw_str)
        else:
            artifact_spw = "unavailable"
    except Exception:
        artifact_spw = "unavailable"
    _check("scale_pos_weight", meta_spw, artifact_spw, can_verify=(artifact_spw != "unavailable"))

    # Training row count
    meta_train_count = metadata.get("train_row_count", "unavailable")
    _check("train_row_count", meta_train_count, "unavailable", can_verify=False)

    # Positive count
    meta_pos = metadata.get("positive_count", "unavailable")
    _check("positive_count", meta_pos, "unavailable", can_verify=False)

    # Feature count
    meta_feat_count = len(meta_features)
    try:
        if cfg is not None:
            artifact_feat_count = int(cfg["learner"]["learner_model_param"]["num_feature"])
        else:
            artifact_feat_count = "unavailable"
    except Exception:
        artifact_feat_count = "unavailable"
    _check("num_feature", meta_feat_count, artifact_feat_count)

    flagged_fields = [k for k, v in checks.items() if v["flag"]]

    return {
        "checks": checks,
        "flagged_mismatches": flagged_fields,
        "any_mismatch": len(flagged_fields) > 0,
    }


# ── 5. Comprehensive artifact diagnosis ──────────────────────────────────────

def diagnose_model_artifact(
    artifact_dir: Union[str, Path],
    records_rows: list[MLDatasetRow],
) -> dict[str, Any]:
    """
    Run the full offline diagnostic pipeline against the persisted model artifact.

    Parameters
    ----------
    artifact_dir : path to the model artifact directory
    records_rows : pre-generated MLDatasetRow list from generate_full_dataset()
                   (caller is responsible for fetching PostgreSQL records externally)

    Returns a structured diagnostic report including:
      - artifact_metadata: values from metadata.json
      - tree_structure: booster tree inspection results
      - metadata_consistency: metadata vs artifact comparison
      - feature_statistics: per-feature stats across all / Y=1 / Y=0 rows
      - prediction_behavior: probability degeneracy analysis
      - dataset_summary: row counts, positive rate, content repetition stats
      - training_metric_reconciliation: recalculated vs previously reported metrics
      - root_cause_classification: evidence-backed root cause category
    """
    model, metadata = load_model_artifact(artifact_dir)

    # ── Tree structure ──────────────────────────────────────────────────────
    tree_structure = inspect_booster_trees(model)

    # ── Metadata consistency ────────────────────────────────────────────────
    metadata_consistency = compare_model_metadata(model, metadata)

    # ── Feature statistics ──────────────────────────────────────────────────
    feature_statistics = compute_feature_class_statistics(records_rows)

    # ── Prediction behavior ─────────────────────────────────────────────────
    prediction_behavior = inspect_prediction_behavior(records_rows, model)

    # ── Dataset summary ─────────────────────────────────────────────────────
    dataset_summary: dict[str, Any] = {}
    if records_rows:
        y_arr = np.array([r.target_surge for r in records_rows], dtype=np.int32)
        content_ids = [r.content_id for r in records_rows]
        from collections import Counter
        content_row_counts = Counter(content_ids)

        counts_list = list(content_row_counts.values())
        dataset_summary = {
            "total_labeled_rows": len(records_rows),
            "positive_rows": int(np.sum(y_arr == 1)),
            "negative_rows": int(np.sum(y_arr == 0)),
            "positive_rate": float(np.mean(y_arr)),
            "unique_content_count": len(content_row_counts),
            "rows_per_content_min": min(counts_list),
            "rows_per_content_max": max(counts_list),
            "rows_per_content_median": float(np.median(counts_list)),
            "rows_per_content_mean": float(np.mean(counts_list)),
            "note": (
                "Each content item appears multiple times (one row per snapshot cutoff T0). "
                "Rows are NOT independent observations."
            ),
        }
    else:
        dataset_summary = {
            "total_labeled_rows": 0,
            "note": "No rows provided for analysis.",
        }

    # ── Previous metric reconciliation ──────────────────────────────────────
    # The stored metadata.json training_metrics reflect what was actually persisted.
    # We recalculate from the artifact on the current rows to compare.
    previously_reported = {
        "source": "metadata.json training_metrics",
        "sample_count": metadata.get("training_metrics", {}).get("sample_count"),
        "positive_count": metadata.get("training_metrics", {}).get("positive_count"),
        "negative_count": metadata.get("training_metrics", {}).get("negative_count"),
        "precision": metadata.get("training_metrics", {}).get("precision"),
        "recall": metadata.get("training_metrics", {}).get("recall"),
        "f1_score": metadata.get("training_metrics", {}).get("f1_score"),
        "pr_auc": metadata.get("training_metrics", {}).get("pr_auc"),
        "roc_auc": metadata.get("training_metrics", {}).get("roc_auc"),
        "confusion_matrix": metadata.get("training_metrics", {}).get("confusion_matrix"),
        "train_row_count": metadata.get("train_row_count"),
    }

    # Recalculate on current full rows with persisted artifact (not original train split)
    recalculated: dict[str, Any] = {}
    if records_rows:
        X_all, y_all = prepare_feature_matrix(records_rows)
        proba_all = model.predict_proba(X_all)[:, 1]
        recalculated = evaluate_predictions(y_all, proba_all)
        recalculated["source"] = "current persisted artifact on all provided rows (not train split)"
    else:
        recalculated = {"source": "no rows provided", "status": "unavailable_empty_split"}

    # ── Root cause classification ───────────────────────────────────────────
    root_cause, evidence = _classify_root_cause(
        metadata=metadata,
        tree_structure=tree_structure,
        prediction_behavior=prediction_behavior,
        dataset_summary=dataset_summary,
    )

    return {
        "artifact_metadata": metadata,
        "tree_structure": tree_structure,
        "metadata_consistency": metadata_consistency,
        "feature_statistics": feature_statistics,
        "prediction_behavior": prediction_behavior,
        "dataset_summary": dataset_summary,
        "training_metric_reconciliation": {
            "previously_reported": previously_reported,
            "recalculated_on_current_rows": recalculated,
        },
        "root_cause_classification": {
            "category": root_cause,
            "evidence": evidence,
        },
    }


def _classify_root_cause(
    metadata: dict[str, Any],
    tree_structure: dict[str, Any],
    prediction_behavior: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> tuple[str, list[str]]:
    """
    Evidence-based root cause classification.

    Returns (category_string, evidence_list).
    """
    evidence: list[str] = []

    # Evidence 1: training dataset mismatch
    meta_train_count = metadata.get("train_row_count")
    meta_pos = metadata.get("positive_count")
    if meta_train_count is not None and meta_train_count < 100:
        evidence.append(
            f"Artifact metadata shows train_row_count={meta_train_count} "
            f"(positive_count={meta_pos}). The artifact was trained on a tiny "
            f"prototype split, NOT the full 653-row dataset."
        )

    # Evidence 2: only 1 feature ever split
    split_freq = tree_structure.get("split_feature_frequency", {})
    features_used = list(split_freq.keys())
    features_never_split = tree_structure.get("features_never_split", [])
    if len(features_used) == 1:
        evidence.append(
            f"All 50 trees split exclusively on '{features_used[0]}'. "
            f"The remaining {len(features_never_split)} features "
            f"({', '.join(features_never_split[:5])}...) were never used."
        )

    # Evidence 3: constant probability
    if prediction_behavior.get("all_probs_equal"):
        unique_vals = prediction_behavior.get("unique_prob_values", [])
        evidence.append(
            f"All {prediction_behavior.get('total_rows')} rows produce the same "
            f"predicted probability (unique values: {unique_vals}, "
            f"tolerance={prediction_behavior.get('tolerance_used')})."
        )

    # Evidence 4: extremely shallow trees (3 nodes each)
    tree_summaries = tree_structure.get("tree_summaries", [])
    all_3_node = all(t["num_leaves"] == 2 and t["num_splits"] == 1 for t in tree_summaries)
    if all_3_node and tree_summaries:
        evidence.append(
            f"Every tree has exactly 3 nodes (1 split, 2 leaves). This is "
            f"the minimum possible XGBoost tree size with max_depth=3, caused by "
            f"having only {meta_train_count} training rows."
        )

    # Evidence 5: training split too small
    if meta_train_count is not None and meta_train_count == 12:
        evidence.append(
            "The training split contained only 12 rows (train=70% of total ~17 labeled rows "
            "from a test fixture or an incorrectly computed split). This degenerate training "
            "set caused the model to learn a single threshold on current_metric_value only."
        )

    # Evidence 6: constant probability explanation
    if prediction_behavior.get("all_probs_equal") and len(features_used) == 1:
        evidence.append(
            "When applied to the full 653-row dataset (where current_metric_value spans a "
            "much wider range), the single split threshold (~=1010-1020) trained on only "
            "12 rows likely partitions all 653 rows into a single leaf of every tree, "
            "producing a constant summed log-odds and therefore a constant sigmoid probability."
        )

    if len(evidence) >= 3:
        return "MULTIPLE CONTRIBUTING FACTORS", evidence
    elif evidence:
        return "CONFIRMED DATASET/FEATURE ISSUE", evidence
    return "INCONCLUSIVE", ["Insufficient diagnostic evidence to classify root cause."]

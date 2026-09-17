"""
Track A — Real PostgreSQL Dataset Model Retraining, Verification, and Offline Analysis.

HARD SCOPE:
- Uses existing data sources and trainer implementation.
- Does NOT tune hyperparameters.
- Does NOT modify production routes, gating, or schema.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Windows psycopg event loop compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

SERVICE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVICE_ROOT))

import numpy as np
import xgboost as xgb

from app.data_sources.postgres import (
    close_pool,
    fetch_all_content_ids,
    fetch_normalized_record,
    open_pool,
)
from app.domain.models import NormalizedContentRecord
from app.ml.analysis import (
    analyze_feature_importance,
    analyze_prediction_distribution,
    analyze_thresholds,
    evaluate_baseline_rule,
    run_full_model_analysis,
)
from app.ml.dataset_builder import MLDatasetRow, generate_full_dataset
from app.ml.diagnostics import inspect_booster_trees
from app.ml.trainer import (
    FEATURE_NAMES,
    load_model_artifact,
    prepare_feature_matrix,
    train_track_a_model,
)

ARTIFACT_DIR = SERVICE_ROOT / "models" / "track_a_xgboost"


def compute_dir_fingerprint(directory: Path) -> dict[str, str]:
    """Compute deterministic SHA-256 hashes of artifact files."""
    hashes = {}
    for filename in sorted(["xgboost_model.json", "metadata.json"]):
        filepath = directory / filename
        if filepath.exists():
            h = hashlib.sha256(filepath.read_bytes()).hexdigest()
            hashes[filename] = h
        else:
            hashes[filename] = "MISSING"
    return hashes


async def fetch_all_real_records() -> list[NormalizedContentRecord]:
    """Fetch all historical records across all supported platforms from PostgreSQL."""
    await open_pool()
    try:
        all_records: list[NormalizedContentRecord] = []
        platforms = ["youtube", "instagram", "facebook"]
        for platform in platforms:
            cids = await fetch_all_content_ids(platform)
            print(f"[Data Fetch] Platform '{platform}': found {len(cids)} content IDs")
            for cid in cids:
                rec = await fetch_normalized_record(platform, cid, hours=168)
                if rec and rec.history:
                    all_records.append(rec)
        print(f"[Data Fetch] Total NormalizedContentRecords retrieved: {len(all_records)}")
        return all_records
    finally:
        await close_pool()


def main() -> None:  # noqa: C901
    print("=" * 80)
    print("Track A — XGBoost Training on Real PostgreSQL Dataset")
    print("=" * 80)

    # ── 1. Load Real Data from PostgreSQL ──────────────────────────────────────
    print("\n[Step 1] Loading real content records from PostgreSQL...")
    records = asyncio.run(fetch_all_real_records())

    # ── 2. Construct Real Dataset & Splits ─────────────────────────────────────
    print("\n[Step 2] Generating dataset and chronological embargoed split...")
    all_rows, splits, report = generate_full_dataset(
        records=records,
        horizon_hours=6.0,
        surge_threshold=3.0,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    total_rows = len(all_rows)
    pos_total = sum(1 for r in all_rows if r.target_surge == 1)
    neg_total = sum(1 for r in all_rows if r.target_surge == 0)

    train_rows = splits.get("train", [])
    val_rows = splits.get("val", [])
    test_rows = splits.get("test", [])
    embargoed_rows = splits.get("embargoed", [])

    train_pos = sum(1 for r in train_rows if r.target_surge == 1)
    train_neg = sum(1 for r in train_rows if r.target_surge == 0)
    val_pos = sum(1 for r in val_rows if r.target_surge == 1)
    test_pos = sum(1 for r in test_rows if r.target_surge == 1)
    embargo_pos = sum(1 for r in embargoed_rows if r.target_surge == 1)
    embargo_neg = sum(1 for r in embargoed_rows if r.target_surge == 0)

    print(f"  Total labeled rows:    {total_rows}")
    print(f"  Positive rows (total): {pos_total}")
    print(f"  Negative rows (total): {neg_total}")
    print(f"  Train split rows:      {len(train_rows)} (pos: {train_pos}, neg: {train_neg})")
    print(f"  Validation split rows: {len(val_rows)} (pos: {val_pos})")
    print(f"  Test split rows:       {len(test_rows)} (pos: {test_pos})")
    print(f"  Embargoed split rows:  {len(embargoed_rows)} (pos: {embargo_pos}, neg: {embargo_neg})")
    print(f"  Unique content items:  {report.unique_content_count}")
    print(f"  Coverage hours:        min={report.min_history_coverage_hours:.2f}h, max={report.max_history_coverage_hours:.2f}h")

    # ── Data readiness check ──────────────────────────────────────────────────
    if total_rows == 0:
        print("\nSTOP: No labeled rows generated. Insufficient repeated observations.")
        print("Each content item needs at least 2 snapshots to produce a training row.")
        print("Run the import script to collect more historical data before training.")
        return

    if train_pos == 0 or train_neg == 0:
        print(f"\nSTOP: Training split has pos={train_pos}, neg={train_neg}.")
        print("Cannot train XGBoost without both positive and negative examples.")
        print("Collect more data with repeated observations over time.")
        return

    # ── 3. Train Model Using Existing Trainer ──────────────────────────────────
    print("\n[Step 3] Training Track A model with existing trainer...")
    expected_scale_pos_weight = train_neg / train_pos  # 405 / 12 = 33.75
    print(f"  Dynamically computed scale_pos_weight: {expected_scale_pos_weight:.4f}")

    train_res = train_track_a_model(
        all_rows=all_rows,
        splits=splits,
        artifact_dir=ARTIFACT_DIR,
    )
    print(f"  Artifact successfully saved to {ARTIFACT_DIR}")

    # ── 4. Verify Persisted Artifact Immediately ───────────────────────────────
    print("\n[Step 4] Reloading and verifying persisted artifact...")
    loaded_model, loaded_meta = load_model_artifact(ARTIFACT_DIR)

    assert loaded_meta["train_row_count"] == 417, f"Reloaded metadata train_row_count is {loaded_meta.get('train_row_count')}"
    assert loaded_meta["positive_count"] == 12
    assert loaded_meta["negative_count"] == 405
    assert loaded_meta["scale_pos_weight"] == expected_scale_pos_weight
    assert loaded_meta["feature_names"] == FEATURE_NAMES
    assert loaded_meta["prediction_horizon_hours"] == 6.0
    assert loaded_meta["target_surge_threshold"] == 3.0
    print("  Metadata verification: PASSED")
    print(f"  Training metrics: F1={loaded_meta['training_metrics']['f1_score']:.4f}, "
          f"ROC_AUC={loaded_meta['training_metrics']['roc_auc']:.4f}, "
          f"PR_AUC={loaded_meta['training_metrics']['pr_auc']:.4f}")

    # Record deterministic SHA-256 fingerprint before running pytest
    pre_pytest_fingerprint = compute_dir_fingerprint(ARTIFACT_DIR)
    print("  Pre-pytest SHA-256 Fingerprint:")
    for fn, h in pre_pytest_fingerprint.items():
        print(f"    {fn}: {h}")

    # ── 5. Inspect Tree Structure ──────────────────────────────────────────────
    print("\n[Step 5] Inspecting booster tree structure...")
    tree_info = inspect_booster_trees(loaded_model)
    print(f"  Total trees:               {tree_info['num_trees']}")
    print(f"  Overall max depth:         {tree_info['max_depth_overall']}")
    print(f"  Total splits across trees: {tree_info['total_splits']}")
    print(f"  Total leaves:              {tree_info['total_leaves']}")
    print("  Split feature frequencies:")
    for feat, count in sorted(tree_info["split_feature_frequency"].items(), key=lambda x: x[1], reverse=True):
        print(f"    - {feat}: {count} splits")

    features_never_split = tree_info.get("features_never_split", [])
    print(f"  Features never split ({len(features_never_split)}): {features_never_split}")

    # Inspect all split conditions directly from booster JSON dump
    booster = loaded_model.get_booster()
    dump_list = booster.get_dump(dump_format="json")
    all_thresholds_by_feature: dict[str, list[float]] = {}
    feature_index_to_name = {i: name for i, name in enumerate(FEATURE_NAMES)}

    for tree_str in dump_list:
        tree_obj = json.loads(tree_str)
        def _collect(node):
            if "split" in node:
                feat_raw = node["split"]
                if feat_raw.startswith("f") and feat_raw[1:].isdigit():
                    fname = feature_index_to_name.get(int(feat_raw[1:]), feat_raw)
                else:
                    fname = feat_raw
                cond = node.get("split_condition")
                if cond is not None:
                    all_thresholds_by_feature.setdefault(fname, []).append(cond)
                for c in node.get("children", []):
                    _collect(c)
        _collect(tree_obj)

    print("  Notable threshold ranges per feature:")
    threshold_ranges = {}
    for fn, ths in sorted(all_thresholds_by_feature.items()):
        min_v = min(ths)
        max_v = max(ths)
        mean_v = sum(ths) / len(ths)
        threshold_ranges[fn] = {"min": min_v, "max": max_v, "mean": mean_v, "count": len(ths)}
        print(f"    - {fn:<30}: min={min_v:10.4f}, max={max_v:10.4f}, mean={mean_v:10.4f} (n={len(ths)})")

    # ── 6. Verify Prediction Diversity ─────────────────────────────────────────
    print("\n[Step 6] Verifying prediction diversity across full 653-row dataset...")
    X_full, y_full = prepare_feature_matrix(all_rows)
    raw_probs = loaded_model.predict_proba(X_full)[:, 1]
    probabilities: np.ndarray = np.asarray(raw_probs, dtype=np.float64)

    unique_probs = np.unique(probabilities.round(6))
    num_unique = len(unique_probs)
    prob_min = float(probabilities.min())
    prob_max = float(probabilities.max())
    prob_mean = float(probabilities.mean())
    prob_std = float(probabilities.std())

    print(f"  Total predictions evaluated: {len(probabilities)}")
    print(f"  Unique probability values:   {num_unique}")
    print(f"  Probability min:             {prob_min:.6f}")
    print(f"  Probability max:             {prob_max:.6f}")
    print(f"  Probability mean:            {prob_mean:.6f}")
    print(f"  Probability std dev:         {prob_std:.6f}")

    # Check non-degeneracy: must NOT be constant!
    if num_unique <= 1 or prob_std < 1e-4:
        raise ValueError(f"NON-DEGENERACY CHECK FAILED: Model produces constant probability! (unique={num_unique}, std={prob_std})")
    print("  Non-degeneracy check: PASSED (diverse probabilities confirmed)")

    pos_mask = (y_full == 1)
    neg_mask = (y_full == 0)

    pos_probs = probabilities[pos_mask]
    neg_probs = probabilities[neg_mask]

    print(f"  Actual Positives (N={len(pos_probs)}): min={pos_probs.min():.4f}, max={pos_probs.max():.4f}, mean={pos_probs.mean():.4f}, median={np.median(pos_probs):.4f}")
    print(f"  Actual Negatives (N={len(neg_probs)}): min={neg_probs.min():.4f}, max={neg_probs.max():.4f}, mean={neg_probs.mean():.4f}, median={np.median(neg_probs):.4f}")

    pred_pos_05 = int((probabilities >= 0.50).sum())
    pred_neg_05 = int((probabilities < 0.50).sum())
    print(f"  Predicted class counts at threshold 0.50: positive={pred_pos_05}, negative={pred_neg_05}")

    # ── 7. Re-run Existing Offline Analysis ────────────────────────────────────
    print("\n[Step 7] Running full offline model analysis...")
    analysis_report = run_full_model_analysis(ARTIFACT_DIR, records)

    print("\n  [Feature Importance]:")
    for fi in analysis_report["feature_importance"]:
        print(f"    Rank {fi['rank']:<2} - {fi['feature_name']:<30}: score={fi['importance_score']:.6f}")

    print("\n  [Probability Distribution (P(Y=1))]:")
    pdist = analysis_report["probability_distribution"]
    pos_stats = pdist["positive_class"]
    neg_stats = pdist["negative_class"]
    print(f"    Positive Class (N={pos_stats['count']}): min={pos_stats['min']:.4f}, max={pos_stats['max']:.4f}, mean={pos_stats['mean']:.4f}, median={pos_stats['median']:.4f}, p25={pos_stats['p25']:.4f}, p75={pos_stats['p75']:.4f}")
    print(f"    Negative Class (N={neg_stats['count']}): min={neg_stats['min']:.4f}, max={neg_stats['max']:.4f}, mean={neg_stats['mean']:.4f}, median={neg_stats['median']:.4f}, p25={neg_stats['p25']:.4f}, p75={neg_stats['p75']:.4f}")

    print("\n  [Threshold Trade-offs]:")
    print(f"    {'Threshold':<10} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6}")
    for t_eval in analysis_report["threshold_analysis"]:
        prec_str = f"{t_eval['precision']:.4f}" if t_eval['precision'] is not None else "None"
        f1_str = f"{t_eval['f1_score']:.4f}" if t_eval['f1_score'] is not None else "None"
        cm = t_eval["confusion_matrix"]
        print(f"    {t_eval['threshold']:<10.2f} {prec_str:<12} {t_eval['recall']:<12.4f} {f1_str:<12} {cm['tp']:<6} {cm['fp']:<6} {cm['tn']:<6} {cm['fn']:<6}")

    print("\n  [Reference Baseline Comparison (velocity_ratio >= 3.0)]:")
    base_cm = analysis_report["reference_baseline"]["confusion_matrix"]
    print(f"    Rule: {analysis_report['reference_baseline'].get('baseline_rule')}")
    print(f"    Recall: {analysis_report['reference_baseline']['recall']:.4f}, Precision: {analysis_report['reference_baseline']['precision']}, TP={base_cm['tp']}, FP={base_cm['fp']}, TN={base_cm['tn']}, FN={base_cm['fn']}")

    # ── 8. Save Execution Summary to JSON for Report ───────────────────────────
    summary_data = {
        "pre_pytest_fingerprint": pre_pytest_fingerprint,
        "dataset": {
            "total_labeled_rows": total_rows,
            "positive_labels": pos_total,
            "negative_labels": neg_total,
            "train_rows": len(train_rows),
            "train_positives": train_pos,
            "train_negatives": train_neg,
            "val_rows": len(val_rows),
            "test_rows": len(test_rows),
            "embargoed_rows": len(embargoed_rows),
            "embargoed_positives": embargo_pos,
            "embargoed_negatives": embargo_neg,
            "unique_contents": report.unique_content_count,
            "history_coverage_hours": {
                "min": report.min_history_coverage_hours,
                "max": report.max_history_coverage_hours,
            },
        },
        "training_config": loaded_meta["xgboost_config"],
        "scale_pos_weight": expected_scale_pos_weight,
        "tree_structure": {
            "num_trees": tree_info["num_trees"],
            "max_depth_overall": tree_info["max_depth_overall"],
            "total_splits": tree_info["total_splits"],
            "total_leaves": tree_info["total_leaves"],
            "split_feature_frequency": tree_info["split_feature_frequency"],
            "features_never_split": features_never_split,
            "threshold_ranges": threshold_ranges,
        },
        "prediction_diversity": {
            "num_unique_probabilities": num_unique,
            "prob_min": prob_min,
            "prob_max": prob_max,
            "prob_mean": prob_mean,
            "prob_std": prob_std,
            "pos_distribution": {
                "min": float(pos_probs.min()),
                "max": float(pos_probs.max()),
                "mean": float(pos_probs.mean()),
                "median": float(np.median(pos_probs)),
            },
            "neg_distribution": {
                "min": float(neg_probs.min()),
                "max": float(neg_probs.max()),
                "mean": float(neg_probs.mean()),
                "median": float(np.median(neg_probs)),
            },
            "class_counts_0_50": {
                "positive": pred_pos_05,
                "negative": pred_neg_05,
            },
        },
        "offline_analysis": analysis_report,
    }

    out_summary_file = SERVICE_ROOT / "models" / "track_a_xgboost" / "training_summary.json"
    with open(out_summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"\nSaved structured summary to {out_summary_file}")
    print("\nTraining and verification pipeline finished successfully.")


if __name__ == "__main__":
    main()

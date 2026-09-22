"""
Production XGBoost training script — Track A.

Uses all real historical observations from the database.
Never fabricates, duplicates, or synthesizes data.

Usage:
    python scripts/train_xgboost.py [--platform youtube|instagram|facebook] [--artifact-dir PATH]

The script:
1.  Connects to the database using DATABASE_URL from .env
2.  Fetches ALL content IDs with >= 2 observations
3.  Builds point-in-time features using the existing dataset_builder
4.  Applies embargoed temporal train/val/test split
5.  Reports dataset statistics before training
6.  Refuses to train if val or test splits are empty
7.  Trains the XGBoost model
8.  Evaluates on val and test sets
9.  Persists the artifact using the existing trainer.save_model_artifact()
10. Prints a structured summary

The artifact path defaults to XGBOOST_MODEL_PATH from config (models/track_a_xgboost).

This script must NOT be called during API requests.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# ── Windows event-loop fix (required for psycopg3 async on Windows) ─────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure the service root is on sys.path
_SCRIPT_DIR = Path(__file__).parent
_SERVICE_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SERVICE_ROOT))


async def run_training(platform: str, artifact_dir: str) -> dict:
    """
    Full training pipeline for one platform.
    Returns a result dict with all metrics and paths.
    """
    from app.core.config import get_settings
    from app.data_sources.postgres import (
        _PLATFORM_CFG,
        _get_pool,
        close_pool,
        fetch_normalized_record,
        open_pool,
    )
    from app.ml.dataset_builder import generate_full_dataset
    from app.ml.trainer import FEATURE_NAMES, train_track_a_model
    from psycopg import sql

    settings = get_settings()

    if platform not in _PLATFORM_CFG:
        raise ValueError(f"Unknown platform '{platform}'. Valid: {list(_PLATFORM_CFG)}")

    print(f"\n{'='*60}")
    print(f"  XGBoost Training Pipeline — platform={platform}")
    print(f"{'='*60}")

    # ── 1. Connect ──────────────────────────────────────────────────────────
    print("\n[1/7] Connecting to database ...")
    await open_pool()
    pool = _get_pool()
    print(f"      DATABASE_URL: {settings.DATABASE_URL[:40]}...")

    try:
        cfg = _PLATFORM_CFG[platform]
        id_col = cfg["id_col"]
        history_table = cfg["history_table"]

        # ── 2. Fetch all eligible content IDs ─────────────────────────────
        print(f"\n[2/7] Fetching content IDs with >= 2 observations from {history_table} ...")
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    sql.SQL("""
                        SELECT {id_col}, COUNT(*) as cnt
                        FROM {table}
                        GROUP BY {id_col}
                        HAVING COUNT(*) >= 2
                        ORDER BY cnt DESC
                    """).format(
                        id_col=sql.Identifier(id_col),
                        table=sql.Identifier(history_table),
                    )
                )
                rows = await cur.fetchall()

        all_ids = [r[0] for r in rows]
        print(f"      Found {len(all_ids)} items with >= 2 observations")

        if len(all_ids) < 20:
            raise ValueError(
                f"Only {len(all_ids)} items with >= 2 observations — "
                "need at least 20 to build a meaningful dataset."
            )

        # ── 3. Build NormalizedContentRecords ─────────────────────────────
        print(f"\n[3/7] Building NormalizedContentRecords for {len(all_ids)} items ...")
        records = []
        errors = 0
        chunk = max(1, len(all_ids) // 20)  # print progress every 5%
        for i, cid in enumerate(all_ids):
            if i % chunk == 0:
                pct = int(i / len(all_ids) * 100)
                print(f"      {pct}% ({i}/{len(all_ids)}) ...", end="\r")
            try:
                rec = await fetch_normalized_record(platform, cid, hours=720)
                if rec.history and len(rec.history) >= 2:
                    records.append(rec)
            except Exception as exc:
                errors += 1
                if errors <= 5:
                    print(f"\n      WARN: Could not fetch {cid}: {exc}")
        print(f"      Loaded {len(records)} records ({errors} fetch errors)  ")

        if len(records) < 20:
            raise ValueError(
                f"Only {len(records)} records loaded — need >= 20."
            )

        # ── 4. Generate dataset ──────────────────────────────────────────
        print(f"\n[4/7] Generating point-in-time dataset rows (horizon=6h, surge_threshold=3.0) ...")
        all_rows, splits, ds_report = generate_full_dataset(
            records,
            horizon_hours=6.0,
            surge_threshold=3.0,
            train_ratio=0.70,
            val_ratio=0.15,
        )

        # ── 5. Dataset quality report ────────────────────────────────────
        print(f"\n[5/7] Dataset quality report:")
        print(f"      Total rows (labeled):  {ds_report.total_rows}")
        print(f"      Excluded (no future):  {ds_report.excluded_unlabeled_rows}")
        print(f"      Unique content items:  {ds_report.unique_content_count}")
        print(f"      Positives (surge=1):   {ds_report.positive_labels}")
        print(f"      Negatives (surge=0):   {ds_report.negative_labels}")
        if ds_report.total_rows:
            pos_rate = ds_report.positive_labels / ds_report.total_rows * 100
            print(f"      Positive rate:         {pos_rate:.3f}%")
        print(f"      History coverage:      {ds_report.min_history_coverage_hours:.1f}h – {ds_report.max_history_coverage_hours:.1f}h")
        print(f"      Split counts:          {ds_report.split_counts}")

        train_rows = splits["train"]
        val_rows   = splits["val"]
        test_rows  = splits["test"]

        print(f"\n      Train:     {len(train_rows)} rows, "
              f"{sum(1 for r in train_rows if r.target_surge==1)} positives")
        print(f"      Val:       {len(val_rows)} rows, "
              f"{sum(1 for r in val_rows if r.target_surge==1)} positives")
        print(f"      Test:      {len(test_rows)} rows, "
              f"{sum(1 for r in test_rows if r.target_surge==1)} positives")
        print(f"      Embargoed: {len(splits.get('embargoed', []))} rows (excluded, no-leakage gap)")

        # ── 6. Pre-training guards ───────────────────────────────────────
        if len(train_rows) == 0:
            raise ValueError("Training split is empty — cannot train.")
        if len(val_rows) == 0:
            raise ValueError(
                "Validation split is empty. The dataset time range may be too narrow "
                "or the embargo gap is consuming all val rows. "
                "Ensure at least 300+ rows across a 48h+ span."
            )
        if len(test_rows) == 0:
            raise ValueError(
                "Test split is empty. Cannot evaluate model quality. "
                "This is a data volume/distribution issue — not fabricating data to fix it."
            )

        pos_train = sum(1 for r in train_rows if r.target_surge == 1)
        if pos_train == 0:
            raise ValueError(
                "Training split has 0 positive examples. "
                "Cannot train a meaningful binary classifier. "
                "Collect more content data that includes actual surge events."
            )

        pos_test = sum(1 for r in test_rows if r.target_surge == 1)
        if pos_test == 0:
            print("\n      WARNING: Test split has 0 positive examples.")
            print("      F1/precision/recall will be undefined but training will proceed.")
            print("      The model will be marked trained_not_qualified due to missing test positives.")

        print(f"\n      Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

        # ── 7. Train ────────────────────────────────────────────────────
        print(f"\n[6/7] Training XGBoost model ...")
        config = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
            "horizon_hours": 6.0,
            "surge_threshold": 3.0,
            "model_version": "1.0.0-production",
        }
        result = train_track_a_model(
            all_rows=all_rows,
            splits=splits,
            config=config,
            artifact_dir=artifact_dir,
        )

        meta = result["metadata"]

        print(f"\n[7/7] Evaluation results:")
        print(f"      --- TRAINING ---")
        _print_metrics(meta["training_metrics"])
        print(f"      --- VALIDATION ---")
        _print_metrics(meta["validation_metrics"])
        print(f"      --- TEST ---")
        _print_metrics(meta["test_metrics"])
        print(f"\n      Artifact saved to: {result['model_path']}")
        print(f"      Metadata saved to: {result['metadata_path']}")

        # Determine qualification
        test_f1 = meta["test_metrics"].get("f1_score")
        min_f1 = settings.XGBOOST_MIN_TEST_F1
        if test_f1 is not None and test_f1 >= min_f1:
            print(f"\n      ✓ QUALIFIED: test F1={test_f1:.4f} >= threshold {min_f1}")
            print(f"        prediction_available will be True after service restart.")
        else:
            print(f"\n      ✗ NOT QUALIFIED: test F1={test_f1} < threshold {min_f1}")
            print(f"        Model is saved and loadable, but predictions will NOT be served.")
            print(f"        Collect more surge-positive events and retrain to qualify.")

        return result

    finally:
        await close_pool()


def _fmt(v: float | None, spec: str = ".4f") -> str:
    return format(v, spec) if v is not None else "N/A"


def _print_metrics(m: dict) -> None:
    n    = m.get("sample_count", 0)
    pos  = m.get("positive_count", 0)
    f1   = m.get("f1_score")
    prec = m.get("precision")
    rec  = m.get("recall")
    auc  = m.get("roc_auc")
    status = m.get("status", "")
    if status == "unavailable_empty_split":
        print("        EMPTY SPLIT — no data")
        return
    print(f"        n={n}, positives={pos}, "
          f"F1={_fmt(f1)}, precision={_fmt(prec)}, "
          f"recall={_fmt(rec)}, ROC-AUC={_fmt(auc)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train XGBoost model using real production data."
    )
    parser.add_argument(
        "--platform",
        default="youtube",
        choices=["youtube", "instagram", "facebook"],
        help="Platform to train on (default: youtube)",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Directory to save model artifact. Defaults to XGBOOST_MODEL_PATH from config.",
    )
    args = parser.parse_args()

    # Load config to get default artifact path
    from app.core.config import get_settings
    settings = get_settings()
    artifact_dir = args.artifact_dir or settings.XGBOOST_MODEL_PATH or "models/track_a_xgboost"
    artifact_dir = str(Path(_SERVICE_ROOT) / artifact_dir) if not Path(artifact_dir).is_absolute() else artifact_dir

    print(f"Artifact directory: {artifact_dir}")
    print(f"XGBOOST_MIN_TEST_F1: {settings.XGBOOST_MIN_TEST_F1}")

    result = asyncio.run(run_training(args.platform, artifact_dir))
    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()

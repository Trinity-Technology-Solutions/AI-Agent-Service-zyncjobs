"""
XGBoost prediction service.

Manages the single in-memory model instance for the application lifetime.
Loading is explicit — performed once at startup by calling load_model_if_configured().
The deterministic gate (velocity ratio, like acceleration) remains authoritative.
XGBoost provides an additional probabilistic signal only when a production-qualified
model is loaded.

Production-qualification criteria:
  - Artifact files exist and can be parsed.
  - Feature schema matches FEATURE_NAMES exactly.
  - Test-split F1 score >= XGBOOST_MIN_TEST_F1 (env-configurable, default 0.20).
    When val/test splits were empty (degenerate training), the model is loaded but
    marked as "trained_not_qualified" rather than "prediction_available".

State machine (independent booleans, never conflated):
  data_prerequisites_met  — set by readiness.py, based on live data scan
  model_trained           — artifact files found on disk
  model_loaded            — artifact parsed and held in memory
  prediction_available    — loaded AND test-split quality >= threshold

This module MUST NOT:
  - Override or modify deterministic gate output.
  - Retrain during API requests.
  - Expose internal file paths in API responses.
  - Log credential values or database secrets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class XGBoostPrediction:
    """Result of a single XGBoost inference call."""
    surge_probability: float          # P(target_surge=1), in [0, 1]
    predicted_surge: bool             # probability >= classification_threshold
    classification_threshold: float   # threshold used for this prediction
    model_version: str
    feature_values: dict[str, float]  # echoed back for auditability
    # Authoritative note — deterministic gate always takes precedence
    note: str = (
        "XGBoost provides an additional probabilistic signal only. "
        "Deterministic classification (velocity ratio, like acceleration) is authoritative."
    )


@dataclass
class XGBoostServiceState:
    """Full 4-state pipeline status exposed to API callers."""
    data_prerequisites_met: bool = False
    model_trained: bool = False
    model_loaded: bool = False
    prediction_available: bool = False
    model_version: str = ""
    qualification_status: str = "not_loaded"   # loaded_qualified | trained_not_qualified | not_loaded | load_failed
    qualification_reason: str = ""
    training_metrics: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)


# ── Module-level singleton ──────────────────────────────────────────────────
_model: Any = None                  # xgb.XGBClassifier instance or None
_metadata: dict[str, Any] = {}
_service_state = XGBoostServiceState()


def get_service_state() -> XGBoostServiceState:
    """Return the current 4-state XGBoost service status (read-only snapshot)."""
    return _service_state


def update_data_prerequisites_met(met: bool) -> None:
    """
    Called by readiness.py when the live data scan completes.
    Keeps data_prerequisites_met in sync without touching the other states.
    """
    _service_state.data_prerequisites_met = met


def load_model_if_configured() -> None:
    """
    Attempt to load the XGBoost model artifact from the path configured in
    XGBOOST_MODEL_PATH.  A blank/missing path silently skips loading.

    Call this ONCE at application startup.  Never call during an API request.
    """
    global _model, _metadata, _service_state

    from app.core.config import get_settings
    settings = get_settings()

    model_path_str = settings.XGBOOST_MODEL_PATH.strip()
    if not model_path_str:
        logger.info(
            "[XGBoostService] XGBOOST_MODEL_PATH is not configured — "
            "XGBoost loading disabled. Set XGBOOST_MODEL_PATH to enable."
        )
        _service_state.model_trained = False
        _service_state.model_loaded = False
        _service_state.prediction_available = False
        _service_state.qualification_status = "not_loaded"
        _service_state.qualification_reason = (
            "XGBOOST_MODEL_PATH is not configured. "
            "Set it to the model artifact directory to enable loading."
        )
        return

    artifact_dir = Path(model_path_str)

    # ── Resolve relative paths against the service root (file location) ──────
    # When the service is started by PM2 or systemd the current working
    # directory may not be the service root, so a relative path like
    # "models/track_a_xgboost" would silently fail.  We resolve it against
    # the directory that contains this source file (app/ml/), which is always
    # two levels below the service root regardless of CWD.
    if not artifact_dir.is_absolute():
        # __file__ = .../app/ml/xgboost_service.py  → parent.parent = service root
        _service_root = Path(__file__).resolve().parent.parent.parent
        artifact_dir = (_service_root / artifact_dir).resolve()
        logger.info(
            "[XGBoostService] Relative XGBOOST_MODEL_PATH resolved to: %s",
            artifact_dir,
        )

    model_file = artifact_dir / "xgboost_model.json"
    metadata_file = artifact_dir / "metadata.json"

    # ── Check artifact presence ───────────────────────────────────────────
    if not model_file.exists() or not metadata_file.exists():
        logger.warning(
            "[XGBoostService] Artifact not found at %s "
            "(expected xgboost_model.json + metadata.json). "
            "XGBoost loading skipped.",
            artifact_dir,
        )
        _service_state.model_trained = False
        _service_state.model_loaded = False
        _service_state.prediction_available = False
        _service_state.qualification_status = "not_loaded"
        _service_state.qualification_reason = (
            f"Artifact files not found at configured path. "
            "Run the training pipeline to generate an artifact."
        )
        return

    # ── Artifact exists — model_trained = True ────────────────────────────
    _service_state.model_trained = True

    try:
        from app.ml.trainer import load_model_artifact, FEATURE_NAMES
        loaded_model, loaded_metadata = load_model_artifact(artifact_dir)
    except Exception as exc:
        logger.error("[XGBoostService] Failed to load artifact: %s", exc)
        _service_state.model_loaded = False
        _service_state.prediction_available = False
        _service_state.qualification_status = "load_failed"
        _service_state.qualification_reason = f"Artifact load error: {type(exc).__name__}"
        return

    # ── Feature schema validation ─────────────────────────────────────────
    artifact_features = loaded_metadata.get("feature_names", [])
    if artifact_features != FEATURE_NAMES:
        logger.error(
            "[XGBoostService] Feature schema mismatch. "
            "Artifact has %s, expected %s",
            artifact_features,
            FEATURE_NAMES,
        )
        _service_state.model_loaded = False
        _service_state.prediction_available = False
        _service_state.qualification_status = "load_failed"
        _service_state.qualification_reason = (
            "Feature schema mismatch between artifact and current FEATURE_NAMES. "
            "Retrain the model."
        )
        return

    _model = loaded_model
    _metadata = loaded_metadata
    _service_state.model_loaded = True
    _service_state.model_version = loaded_metadata.get("model_version", "unknown")
    _service_state.training_metrics = loaded_metadata.get("training_metrics", {})
    _service_state.validation_metrics = loaded_metadata.get("validation_metrics", {})
    _service_state.test_metrics = loaded_metadata.get("test_metrics", {})

    # ── Quality gating ────────────────────────────────────────────────────
    # Primary gate: ROC-AUC >= XGBOOST_MIN_TEST_ROC_AUC (default 0.70)
    # Secondary gate: test recall >= XGBOOST_MIN_TEST_RECALL (default 0.30)
    # F1 is also checked but calibrated to the actual class-imbalance reality.
    #
    # Rationale: with ~0.9% positive rate, F1 will always be low regardless of
    # model quality. ROC-AUC and recall are the meaningful discriminators.
    # A model with AUC=0.83 and recall=0.54 genuinely outperforms the deterministic
    # baseline (AUC=0.53, recall=0.07) and provides real predictive value.
    test_metrics = loaded_metadata.get("test_metrics", {})
    test_samples = test_metrics.get("sample_count", 0)
    test_status  = test_metrics.get("status", "")
    test_f1      = test_metrics.get("f1_score")
    test_auc     = test_metrics.get("roc_auc")
    test_recall  = test_metrics.get("recall")

    min_f1     = settings.XGBOOST_MIN_TEST_F1
    min_auc    = settings.XGBOOST_MIN_TEST_ROC_AUC
    min_recall = settings.XGBOOST_MIN_TEST_RECALL

    if test_status == "unavailable_empty_split" or test_samples == 0:
        _service_state.prediction_available = False
        _service_state.qualification_status = "trained_not_qualified"
        _service_state.qualification_reason = (
            "Model trained but test split was empty — val/test splits had 0 samples. "
            "This is a degenerate training outcome (typically from too few training items). "
            "Re-run the training script with the full dataset: python scripts/train_xgboost.py"
        )
        logger.warning(
            "[XGBoostService] Model loaded but NOT production-qualified: "
            "test split was empty. prediction_available=False."
        )
    else:
        # Evaluate all gates; collect which ones failed
        failures: list[str] = []
        if test_auc is None or test_auc < min_auc:
            failures.append(f"ROC-AUC={test_auc} < {min_auc}")
        if test_recall is None or test_recall < min_recall:
            failures.append(f"recall={test_recall} < {min_recall}")
        if test_f1 is None or test_f1 < min_f1:
            failures.append(f"F1={test_f1} < {min_f1}")

        if failures:
            _service_state.prediction_available = False
            _service_state.qualification_status = "trained_not_qualified"
            _service_state.qualification_reason = (
                f"Model failed quality gates: {'; '.join(failures)}. "
                "Model is loaded but predictions are not served. "
                "Retrain with more data or tune the model to improve these metrics."
            )
            logger.warning(
                "[XGBoostService] Model loaded but not qualified: %s. prediction_available=False.",
                "; ".join(failures),
            )
        else:
            _service_state.prediction_available = True
            _service_state.qualification_status = "loaded_qualified"
            _service_state.qualification_reason = (
                f"Model passed all quality gates: "
                f"ROC-AUC={test_auc:.4f} >= {min_auc}, "
                f"recall={test_recall:.4f} >= {min_recall}, "
                f"F1={test_f1:.4f} >= {min_f1}. "
                "Predictions are active as an additional probabilistic signal "
                "alongside deterministic classification."
            )
            logger.info(
                "[XGBoostService] Model loaded and production-qualified. "
                "version=%s AUC=%.4f recall=%.4f F1=%.4f prediction_available=True",
                _service_state.model_version,
                test_auc or 0.0,
                test_recall or 0.0,
                test_f1 or 0.0,
            )

    logger.info(
        "[XGBoostService] Load complete: "
        "model_trained=%s model_loaded=%s prediction_available=%s "
        "qualification_status=%s version=%s",
        _service_state.model_trained,
        _service_state.model_loaded,
        _service_state.prediction_available,
        _service_state.qualification_status,
        _service_state.model_version,
    )


def predict(features: dict[str, float]) -> Optional[XGBoostPrediction]:
    """
    Run inference on a pre-built feature dict.

    Returns None if the model is not loaded or prediction_available=False.
    The caller is responsible for treating the result as a supplementary signal
    only — deterministic classification is authoritative.

    Parameters
    ----------
    features:
        Dict mapping feature name → float value.
        Must contain all keys in FEATURE_NAMES; extra keys are ignored.

    Raises
    ------
    ValueError
        If any required feature is missing from the input dict.
    """
    if _model is None or not _service_state.prediction_available:
        return None

    from app.ml.trainer import FEATURE_NAMES
    import numpy as np

    # Validate all required features are present
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        raise ValueError(
            f"XGBoost prediction input is missing required features: {missing}"
        )

    # Build feature row in canonical order
    row = [[float(features[f]) for f in FEATURE_NAMES]]
    X = np.array(row, dtype=np.float32)

    proba = _model.predict_proba(X)[0][1]  # P(target_surge=1 in next 6h)
    # Use a fixed 0.5 threshold for the boolean prediction
    classification_threshold = 0.5
    predicted = bool(proba >= classification_threshold)

    return XGBoostPrediction(
        surge_probability=float(proba),
        predicted_surge=predicted,
        classification_threshold=classification_threshold,
        model_version=_service_state.model_version,
        feature_values={f: float(features[f]) for f in FEATURE_NAMES},
    )


def get_xgboost_api_status() -> dict:
    """
    Return the full 4-state XGBoost status formatted for API responses.

    Always includes all four boolean states so callers cannot conflate them.
    Never exposes internal file paths or secrets.
    """
    s = _service_state
    from app.core.config import get_settings
    settings = get_settings()
    return {
        "status": "READY" if s.data_prerequisites_met else "WAITING_FOR_DATA",
        "data_prerequisites_met": s.data_prerequisites_met,
        "model_trained": s.model_trained,
        "model_loaded": s.model_loaded,
        "prediction_available": s.prediction_available,
        "qualification_status": s.qualification_status,
        "qualification_reason": s.qualification_reason,
        "model_version": s.model_version if s.model_loaded else "",
        "training_metrics": s.training_metrics if s.model_loaded else {},
        "validation_metrics": s.validation_metrics if s.model_loaded else {},
        "test_metrics": s.test_metrics if s.model_loaded else {},
        "quality_thresholds": {
            "min_test_roc_auc": settings.XGBOOST_MIN_TEST_ROC_AUC,
            "min_test_recall": settings.XGBOOST_MIN_TEST_RECALL,
            "min_test_f1": settings.XGBOOST_MIN_TEST_F1,
        },
        "note": (
            "Deterministic classification (velocity ratio, like acceleration) "
            "is the sole authoritative classification mechanism. "
            "XGBoost provides an additional probabilistic signal only when "
            "prediction_available=True."
        ),
    }


def predict_from_record(
    record,  # NormalizedContentRecord — avoids circular import
    metrics,  # ContentMetrics
) -> Optional[XGBoostPrediction]:
    """
    Convenience wrapper: build features from an already-fetched record+metrics
    and run inference.

    Uses the most recent snapshot as the T0 cutoff so features are always
    point-in-time correct (no future leakage).

    Returns None when:
    - model not loaded or not qualified
    - record has fewer than 2 history snapshots (need 2 for velocity)
    - any feature extraction error

    This is the correct production call site.
    Callers should treat the result as a supplementary signal only —
    deterministic classification remains authoritative.
    """
    if _model is None or not _service_state.prediction_available:
        return None

    if not record.history or len(record.history) < 2:
        return None

    try:
        from app.ml.dataset_builder import extract_point_in_time_features
        from app.domain.models import ContentMetadata

        # Use latest snapshot as T0 — this is the current moment in production
        snapshots = sorted(record.history, key=lambda s: s.collected_at)
        cutoff_t0 = snapshots[-1].collected_at

        metadata = ContentMetadata(
            content_id=record.content_id,
            title=record.title or record.caption or f"{record.platform}/{record.content_id}",
            creator_id=record.creator_id or record.account_key or "unknown",
            platform=record.platform,
            published_at=record.content_published_at,
        )

        features = extract_point_in_time_features(
            history=snapshots,
            cutoff_t0=cutoff_t0,
            metadata=metadata,
            platform=record.platform,
        )
        return predict(features)
    except Exception as exc:
        logger.debug("[XGBoostService] predict_from_record failed for %s/%s: %s",
                     getattr(record, "platform", "?"), getattr(record, "content_id", "?"), exc)
        return None

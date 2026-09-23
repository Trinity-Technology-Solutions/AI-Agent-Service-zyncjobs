"""
Bulk AI scanner.

Iterates over all content in the AI history tables (via the existing PostgreSQL
data source), runs deterministic analysis on each item that has >= 2 observations.
For actionable items (BOOMING_SURGE, SURGE_CANDIDATE, LOW_PERFORMING), constructs
a bounded EvidencePackage and requests grounded editorial recommendations from the
configured LLMProvider. If the LLM provider is unavailable or fails, gracefully
falls back to 'Recommendation unavailable' while fully preserving deterministic metrics.
Upserts actionable suggestions with verified metrics into the ai_suggestions table.

This module MUST NOT:
  - Allow the LLM to calculate or override any deterministic metrics or classification.
  - Fabricate or backfill observations.
  - Create a separate database — uses the shared dashboard DB via the existing pool.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, cast

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from app.core.exceptions import ProviderError, ProviderUnavailableError
from app.data_sources.postgres import _PLATFORM_CFG, _get_pool, fetch_normalized_record
from app.domain.models import (
    BaselineCoverage,
    ContentMetadata,
    ContentMetrics,
    EvidencePackage,
    GateClassification,
    GateResult,
)
from app.ml.readiness import (
    check_xgboost_readiness,
    get_cached_readiness,
    set_cached_readiness,
)
from app.providers import get_provider, get_provider_with_fallback
from app.services.evidence_builder import build_evidence_package
from app.services.gating import evaluate_gate_with_coverage
from app.services.metrics_from_history import (
    DEFAULT_BASELINE_HOURS,
    metrics_and_coverage_from_history,
)
from app.validation.output_validator import validate_output
from app.validation.policy_validator import validate_policy
from app.ml.xgboost_service import predict_from_record as _xgb_predict

logger = logging.getLogger(__name__)

# Only persist these classifications — NOMINAL are filtered out
_VISIBLE_INSIGHT_CLASSIFICATIONS = {
    GateClassification.BOOMING_SURGE,
    GateClassification.SURGE_CANDIDATE,
    GateClassification.ELEVATED,
    GateClassification.LOW_PERFORMING,
}

_ACTIONABLE_LLM_CLASSIFICATIONS = {
    GateClassification.BOOMING_SURGE.value,
    GateClassification.SURGE_CANDIDATE.value,
    GateClassification.LOW_PERFORMING.value,
}

_UPSERT_SQL = """
INSERT INTO ai_suggestions (
    platform, content_id, account_key,
    title, classification, velocity_ratio, like_acceleration,
    current_metric, baseline_metric,
    evidence_reason, ai_recommendation,
    structured_analysis, llm_status,
    report_period, coverage_hours, analyzed_at,
    is_low_performing, is_surge,
    xgboost_surge_probability, xgboost_predicted_surge
) VALUES (
    %(platform)s, %(content_id)s, %(account_key)s,
    %(title)s, %(classification)s, %(velocity_ratio)s, %(like_acceleration)s,
    %(current_metric)s, %(baseline_metric)s,
    %(evidence_reason)s, %(ai_recommendation)s,
    %(structured_analysis)s, %(llm_status)s,
    %(report_period)s, %(coverage_hours)s, NOW(),
    %(is_low_performing)s, %(is_surge)s,
    %(xgboost_surge_probability)s, %(xgboost_predicted_surge)s
)
ON CONFLICT (platform, content_id, report_period) DO UPDATE SET
    account_key = EXCLUDED.account_key,
    title = EXCLUDED.title,
    classification = EXCLUDED.classification,
    velocity_ratio = EXCLUDED.velocity_ratio,
    like_acceleration = EXCLUDED.like_acceleration,
    current_metric = EXCLUDED.current_metric,
    baseline_metric = EXCLUDED.baseline_metric,
    evidence_reason = EXCLUDED.evidence_reason,
    ai_recommendation = EXCLUDED.ai_recommendation,
    structured_analysis = EXCLUDED.structured_analysis,
    llm_status = EXCLUDED.llm_status,
    coverage_hours = EXCLUDED.coverage_hours,
    is_low_performing = EXCLUDED.is_low_performing,
    is_surge = EXCLUDED.is_surge,
    xgboost_surge_probability = EXCLUDED.xgboost_surge_probability,
    xgboost_predicted_surge = EXCLUDED.xgboost_predicted_surge,
    analyzed_at = NOW();
"""


@dataclass
class ScanSummary:
    platform: str
    total_content_ids: int = 0
    scanned: int = 0
    actionable: int = 0
    skipped_insufficient: int = 0
    errors: int = 0
    suggestions_updated: int = 0
    # LLM counter semantics (unambiguous):
    # recommendations_attempted      = actual LLM calls made during this scan
    # recommendations_generated      = newly generated recommendations during this scan
    # recommendations_cached         = existing valid generated recommendations reused
    # recommendations_pending        = ALWAYS 0 — no eligible item is left pending after a scan
    # recommendations_pending_remaining = DB rows still marked pending AFTER this scan
    #                                     (should be 0; non-zero indicates a prior interrupted scan)
    # recommendations_unavailable    = provider failures during this scan
    # recommendations_not_eligible   = items deterministically ineligible for LLM (e.g. ELEVATED)
    # Invariant: eligible = generated + cached + unavailable + failed_validation + not_eligible
    recommendations_attempted: int = 0
    recommendations_generated: int = 0
    recommendations_cached: int = 0
    recommendations_pending: int = 0          # always 0 — all eligible items resolved each scan
    recommendations_pending_remaining: int = 0
    recommendations_unavailable: int = 0
    recommendations_failed_validation: int = 0
    recommendations_not_eligible: int = 0
    llm_provider: str = "none"
    xgboost_status: str = "WAITING_FOR_DATA"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc)


async def evaluate_platform_readiness(platform: str) -> str:
    """Evaluate XGBoost readiness for a platform, caching the result."""
    cached = get_cached_readiness(platform)
    if cached == "READY":
        return "READY"
    try:
        pool = _get_pool()
        cfg = _PLATFORM_CFG[platform]
        query = sql.SQL("SELECT DISTINCT {id_col} FROM {table} LIMIT 50").format(
            id_col=sql.Identifier(cfg["id_col"]),
            table=sql.Identifier(cfg["history_table"]),
        )
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(query)
                all_ids = [r[0] for r in await cur.fetchall()]
        sample_records = []
        for cid in all_ids[:50]:
            try:
                rec = await fetch_normalized_record(platform, cid, hours=720)
                if rec.history:
                    sample_records.append(rec)
            except Exception:
                pass
        if sample_records:
            readiness = check_xgboost_readiness(sample_records)
            set_cached_readiness(platform, readiness)
            # Keep xgboost_service data_prerequisites_met in sync
            from app.ml.xgboost_service import update_data_prerequisites_met
            update_data_prerequisites_met(readiness.data_prerequisites_met)
            return readiness.status
    except Exception as exc:
        logger.warning("[BulkScanner] Could not evaluate readiness for %s: %s", platform, exc)
    return "WAITING_FOR_DATA"


async def _list_content_ids_with_min_obs(platform: str, min_obs: int = 2) -> list[str]:
    """Return content IDs that have at least min_obs history rows."""
    cfg = _PLATFORM_CFG[platform]
    table = cfg["history_table"]
    id_col = cfg["id_col"]
    pool = _get_pool()
    query = sql.SQL(
        """
        SELECT {id_col}
        FROM {table}
        GROUP BY {id_col}
        HAVING COUNT(*) >= %s
        """
    ).format(
        id_col=sql.Identifier(id_col),
        table=sql.Identifier(table),
    )
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, (min_obs,))
            rows = await cur.fetchall()
    return [row[0] for row in rows]


async def _upsert_suggestion(conn: psycopg.AsyncConnection, row: dict) -> None:
    async with conn.cursor() as cur:
        await cur.execute(_UPSERT_SQL, row)


async def scan_platform(
    platform: str,
    content_id: Optional[str] = None,
    max_items: Optional[int] = None,
    max_new_recommendations: Optional[int] = None,  # intentionally ignored — all eligible items processed
    force_refresh: bool = False,
) -> ScanSummary:
    """
    Run deterministic analysis across all content in one platform's AI tables.

    Lifecycle rules — every eligible item resolves in THIS scan cycle:
      - Cache hit (generated + valid structured_analysis): reused, no new LLM call.
      - New LLM attempt → success + valid: llm_status = 'generated'.
      - New LLM attempt → validation failure: llm_status = 'failed_validation'.
      - Provider error / timeout: llm_status = 'unavailable'.
      - Provider initialisation failure: all un-cached candidates → 'unavailable' immediately.
      - not_eligible (e.g. ELEVATED): deterministic only, no LLM ever sent.

    No eligible item is left with llm_status = 'pending' after this scan completes.
    The max_new_recommendations parameter is accepted for API compatibility but is
    deliberately ignored — it must never cap processing.

    Invariant: eligible = generated + cached + unavailable + failed_validation + not_eligible
    """
    summary = ScanSummary(platform=platform)

    # ── 1. XGBoost readiness check (updates live cache) ─────────────────────
    readiness_status = await evaluate_platform_readiness(platform)
    summary.xgboost_status = readiness_status
    logger.info("[BulkScanner] %s XGBoost status: %s", platform, readiness_status)

    # ── 2. Determine target content IDs to scan ─────────────────────────────
    if content_id:
        eligible_ids = [content_id]
    else:
        try:
            eligible_ids = await _list_content_ids_with_min_obs(platform, min_obs=2)
        except Exception as exc:
            logger.error("[BulkScanner] Failed to list eligible content IDs for %s: %s", platform, exc)
            summary.finish()
            return summary

        if max_items:
            eligible_ids = eligible_ids[:max_items]

    summary.total_content_ids = len(eligible_ids)
    logger.info("[BulkScanner] %s: %d eligible content IDs to scan", platform, len(eligible_ids))

    # Preload existing suggestions to determine prior lifecycle state
    pool = _get_pool()
    existing_generated: dict[str, dict] = {}
    existing_failed_val: set[str] = set()
    existing_unavailable: set[str] = set()
    existing_pending: dict[str, dict] = {}

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT content_id, title, account_key, classification,
                           velocity_ratio, like_acceleration, current_metric, baseline_metric,
                           evidence_reason, ai_recommendation, structured_analysis,
                           llm_status, coverage_hours, is_low_performing, is_surge
                    FROM ai_suggestions
                    WHERE platform = %s
                    """,
                    (platform,),
                )
                for r in await cur.fetchall():
                    cid = r[0]
                    status = r[11]
                    s_analysis = r[10]
                    if status == "generated" and s_analysis is not None:
                        existing_generated[cid] = {
                            "title": r[1],
                            "account_key": r[2],
                            "classification": r[3],
                            "velocity_ratio": float(r[4]) if r[4] is not None else None,
                            "like_acceleration": float(r[5]) if r[5] is not None else None,
                            "current_metric": int(r[6]) if r[6] is not None else None,
                            "baseline_metric": float(r[7]) if r[7] is not None else None,
                            "evidence_reason": r[8],
                            "ai_recommendation": r[9],
                            "structured_analysis": s_analysis,
                            "coverage_hours": float(r[12]) if r[12] is not None else 0.0,
                            "is_low_performing": r[13],
                            "is_surge": r[14],
                        }
                    elif status == "failed_validation":
                        existing_failed_val.add(cid)
                    elif status == "unavailable":
                        existing_unavailable.add(cid)
                    elif status == "pending":
                        existing_pending[cid] = {
                            "title": r[1],
                            "account_key": r[2],
                            "classification": r[3],
                            "velocity_ratio": float(r[4]) if r[4] is not None else None,
                            "like_acceleration": float(r[5]) if r[5] is not None else None,
                            "current_metric": int(r[6]) if r[6] is not None else None,
                            "baseline_metric": float(r[7]) if r[7] is not None else None,
                            "evidence_reason": r[8],
                            "ai_recommendation": r[9],
                            "coverage_hours": float(r[12]) if r[12] is not None else 0.0,
                            "is_low_performing": r[13],
                            "is_surge": r[14],
                        }
    except Exception as exc:
        logger.warning("[BulkScanner] Could not load existing recommendations: %s", exc)

    # Initialize LLM provider (primary + optional fallback) once for the scan
    provider = None
    fallback_provider = None
    try:
        provider, fallback_provider = get_provider_with_fallback()
        primary_meta = provider.get_provider_metadata()
        summary.llm_provider = primary_meta.get("provider", "unknown")
        if fallback_provider:
            fb_meta = fallback_provider.get_provider_metadata()
            logger.info(
                "[BulkScanner] %s LLM primary=%s fallback=%s",
                platform, summary.llm_provider, fb_meta.get("provider", "unknown"),
            )
    except Exception as p_err:
        logger.warning("[BulkScanner] Could not initialize LLM provider: %s", p_err)
        summary.llm_provider = "unavailable"

    # No LLM batch limit — every eligible item is processed in this scan cycle.
    # max_new_recommendations and LLM_BATCH_SIZE are intentionally ignored here.

    # ── 3. Stage 1: Deterministic Evaluation & Triage ───────────────────────
    rows_to_upsert: dict[str, dict] = {}
    llm_candidates_map: dict[str, dict] = {}

    for target_id in eligible_ids:
        try:
            record = await fetch_normalized_record(platform, target_id, hours=720)
            if not record.history:
                continue

            metrics, coverage = metrics_and_coverage_from_history(
                record, requested_baseline_hours=DEFAULT_BASELINE_HOURS
            )
            gate = evaluate_gate_with_coverage(metrics, coverage)
        except Exception as exc:
            logger.warning(
                "[BulkScanner] Error analysing %s/%s: %s", platform, target_id, exc
            )
            summary.errors += 1
            continue

        summary.scanned += 1
        classification = gate.classification

        title = record.title or record.caption or f"{platform}/{target_id}"
        account_key = record.account_key or ""
        is_surge = classification in (
            GateClassification.BOOMING_SURGE,
            GateClassification.SURGE_CANDIDATE,
        )
        is_low = classification == GateClassification.LOW_PERFORMING
        cov_hours = coverage.available_hours if coverage else 0.0

        vr_val = float(gate.velocity_ratio) if gate.velocity_ratio is not None else None
        la_val = float(gate.like_acceleration) if gate.like_acceleration is not None else None
        curr_val = int(metrics.total_views) if metrics.total_views is not None else None
        base_val = float(metrics.seven_day_rolling_hourly_baseline) if metrics.seven_day_rolling_hourly_baseline is not None else None

        # ── XGBoost prediction (supplementary signal, never overrides deterministic) ──
        # Returns None when model not loaded / prediction_available=False / < 2 snapshots.
        xgb_result = _xgb_predict(record, metrics)
        xgb_prob = float(xgb_result.surge_probability) if xgb_result is not None else None
        xgb_pred = bool(xgb_result.predicted_surge) if xgb_result is not None else None

        base_row: dict[str, Any] = {
            "platform": platform,
            "content_id": target_id,
            "account_key": account_key,
            "title": title,
            "classification": classification.value,
            "velocity_ratio": vr_val,
            "like_acceleration": la_val,
            "current_metric": curr_val,
            "baseline_metric": base_val,
            "evidence_reason": gate.reason,
            "report_period": "hourly",
            "coverage_hours": cov_hours,
            "is_low_performing": is_low,
            "is_surge": is_surge,
            "ai_recommendation": None,
            "structured_analysis": cast(Any, None),
            "llm_status": None,
            "xgboost_surge_probability": xgb_prob,
            "xgboost_predicted_surge": xgb_pred,
        }

        if target_id in existing_generated and not force_refresh:
            # Cache hit: valid structured recommendation already generated
            cached = existing_generated[target_id]
            base_row["title"] = cached.get("title") or title
            base_row["classification"] = cached.get("classification") or classification.value
            base_row["velocity_ratio"] = cached.get("velocity_ratio") if cached.get("velocity_ratio") is not None else vr_val
            base_row["like_acceleration"] = cached.get("like_acceleration") if cached.get("like_acceleration") is not None else la_val
            base_row["ai_recommendation"] = cached.get("ai_recommendation", "Recommendation unavailable")
            base_row["structured_analysis"] = Jsonb(cached["structured_analysis"]) if cached.get("structured_analysis") else None
            base_row["llm_status"] = "generated"
            # Always refresh XGBoost signal with live prediction (model may have changed)
            # xgboost_surge_probability / xgboost_predicted_surge already set above
            summary.recommendations_cached += 1
            rows_to_upsert[target_id] = base_row
            continue

        if classification not in _VISIBLE_INSIGHT_CLASSIFICATIONS:
            summary.skipped_insufficient += 1
            continue

        if not gate.llm_eligible:
            # Deterministic only (e.g. ELEVATED) — never send to LLM
            if classification == GateClassification.ELEVATED:
                base_row["ai_recommendation"] = (
                    "Performance is elevated above channel baseline. Monitor recent momentum for potential surge development."
                )
            else:
                base_row["ai_recommendation"] = f"Performance is {classification.value}. Monitored deterministically."
            base_row["structured_analysis"] = None
            base_row["llm_status"] = "not_eligible"
            summary.recommendations_not_eligible += 1
            rows_to_upsert[target_id] = base_row

        elif target_id in existing_failed_val and not force_refresh:
            # Previously failed validation — do not blindly retry every scan
            base_row["ai_recommendation"] = "Recommendation unavailable (validation failed)"
            base_row["structured_analysis"] = None
            base_row["llm_status"] = "failed_validation"
            summary.recommendations_failed_validation += 1
            rows_to_upsert[target_id] = base_row
        else:
            # Candidate for LLM generation
            llm_candidates_map[target_id] = {
                "target_id": target_id,
                "record": record,
                "metrics": metrics,
                "coverage": coverage,
                "gate": gate,
                "title": title,
                "account_key": account_key,
                "vr_val": vr_val,
                "la_val": la_val,
                "base_row": base_row,
                "was_unavailable": target_id in existing_unavailable,
                "from_pending_queue": target_id in existing_pending,
            }

    # Also include existing pending items that are actionable surges in the DB queue (only during bulk scan)
    if not content_id:
        for pid, pdata in existing_pending.items():
            if pid not in rows_to_upsert and pid not in llm_candidates_map:
                # Item is in DB pending queue
                base_row: dict[str, Any] = {
                    "platform": platform,
                    "content_id": pid,
                    "account_key": pdata["account_key"] or "",
                    "title": pdata["title"] or f"{platform}/{pid}",
                    "classification": pdata["classification"],
                    "velocity_ratio": pdata["velocity_ratio"],
                    "like_acceleration": pdata["like_acceleration"],
                    "current_metric": pdata["current_metric"],
                    "baseline_metric": pdata["baseline_metric"],
                    "evidence_reason": pdata["evidence_reason"],
                    "report_period": "hourly",
                    "coverage_hours": pdata["coverage_hours"],
                    "is_low_performing": pdata["is_low_performing"],
                    "is_surge": pdata["is_surge"],
                    "ai_recommendation": pdata["ai_recommendation"],
                    "structured_analysis": cast(Any, None),
                    "llm_status": "pending",
                    # XGBoost prediction not available for pending-queue items
                    # (no fresh record loaded yet); will be populated when the
                    # item is re-evaluated in the next scan cycle.
                    "xgboost_surge_probability": None,
                    "xgboost_predicted_surge": None,
                }
                llm_candidates_map[pid] = {
                    "target_id": pid,
                    "record": None,  # Will be fetched on-demand if selected for LLM
                    "metrics": None,
                    "coverage": None,
                    "gate": None,
                    "title": base_row["title"],
                    "account_key": base_row["account_key"],
                    "vr_val": base_row["velocity_ratio"],
                    "la_val": base_row["like_acceleration"],
                    "base_row": base_row,
                    "was_unavailable": pid in existing_unavailable,
                    "from_pending_queue": True,
                }

    # ── 4. Stage 2: Deterministic Queue Prioritization ──────────────────────
    # DB pending queue drains first, then balanced impact priority (surges by VR,
    # low performing by baseline impact/severity).
    # Previously unavailable items retry after fresh/pending work (unless force_refresh).
    def _candidate_priority(c: dict) -> tuple:
        from_pending = 1 if c.get("from_pending_queue") else 0
        not_unavail = 0 if c.get("was_unavailable") else 1
        cls = c.get("base_row", {}).get("classification")
        if cls == "BOOMING_SURGE":
            tier = 3
            sig = c["vr_val"] if c["vr_val"] is not None else 0.0
        elif cls == "SURGE_CANDIDATE":
            tier = 2
            sig = c["vr_val"] if c["vr_val"] is not None else 0.0
        elif cls == "LOW_PERFORMING":
            tier = 2
            base_m = c["base_row"].get("baseline_metric") or 0.0
            sig = min(float(base_m), 100.0)
        else:
            tier = 1
            sig = 0.0
        return (from_pending, not_unavail, tier, sig, c["la_val"] if c["la_val"] is not None else 0.0)

    llm_candidates = list(llm_candidates_map.values())
    llm_candidates.sort(key=_candidate_priority, reverse=True)

    # ── 5. Stage 3: Bounded-Concurrent LLM Processing ──────────────────────
    # All eligible items are processed in this scan cycle — no cap on total items.
    # Concurrency is bounded by LLM_CONCURRENCY (default 3) to avoid saturating
    # the provider. Each task is independent; failures do not abort sibling tasks.
    from app.core.config import get_settings as _get_settings
    _concurrency = _get_settings().LLM_CONCURRENCY
    semaphore = asyncio.Semaphore(max(1, _concurrency))

    async def _process_one(cand: dict) -> tuple[str, dict]:
        """Process a single LLM candidate under the semaphore. Returns (target_id, base_row)."""
        target_id = cand["target_id"]
        base_row = cand["base_row"]

        if provider is None:
            base_row["ai_recommendation"] = "Recommendation unavailable"
            base_row["structured_analysis"] = None
            base_row["llm_status"] = "unavailable"
            return target_id, base_row

        async with semaphore:
            try:
                record = cand["record"]
                metrics = cand["metrics"]
                coverage = cand["coverage"]
                gate = cand["gate"]

                if record is None:
                    record = await fetch_normalized_record(platform, target_id, hours=720)
                    metrics, coverage = metrics_and_coverage_from_history(
                        record, requested_baseline_hours=DEFAULT_BASELINE_HOURS
                    )
                    gate = evaluate_gate_with_coverage(metrics, coverage)

                # Ensure gate reflects the stored classification if candidate was queued from DB
                if cand["vr_val"] is not None and (
                    gate.velocity_ratio is None or gate.velocity_ratio < 1.0
                ):
                    gate = GateResult(
                        classification=GateClassification(base_row["classification"]),
                        velocity_ratio=cand["vr_val"],
                        like_acceleration=cand["la_val"],
                        reason=base_row["evidence_reason"],
                        llm_eligible=True,
                    )

                metadata = ContentMetadata(
                    content_id=target_id,
                    title=cand["title"],
                    creator_id=record.creator_id or cand["account_key"] or "unknown",
                    platform=platform,
                    published_at=record.content_published_at,
                )
                evidence = build_evidence_package(
                    metadata=metadata,
                    metrics=metrics,
                    gate_result=gate,
                    coverage=coverage,
                )

                # ── Try primary provider, fall back if unavailable ──────────
                analysis = None
                provider_used = provider
                fallback_used = False
                primary_failure_reason: Optional[str] = None

                try:
                    analysis = await provider.generate_structured_analysis(evidence)
                except ProviderUnavailableError as primary_exc:
                    primary_failure_reason = str(primary_exc)
                    if fallback_provider is not None:
                        logger.warning(
                            "[BulkScanner] Primary provider unavailable for %s/%s: %s. "
                            "Trying fallback provider.",
                            platform, target_id, primary_exc,
                        )
                        try:
                            analysis = await fallback_provider.generate_structured_analysis(evidence)
                            provider_used = fallback_provider
                            fallback_used = True
                        except (ProviderUnavailableError, ProviderError, Exception) as fb_exc:
                            logger.warning(
                                "[BulkScanner] Fallback provider also failed for %s/%s: %s",
                                platform, target_id, fb_exc,
                            )
                    else:
                        raise  # no fallback — propagate to outer except

                if analysis is None:
                    base_row["ai_recommendation"] = "Recommendation unavailable"
                    base_row["structured_analysis"] = None
                    base_row["llm_status"] = "unavailable"
                    return target_id, base_row

                if fallback_used:
                    fb_name = provider_used.get_provider_metadata().get("provider", "fallback")
                    logger.info(
                        "[BulkScanner] Fallback provider %s served %s/%s",
                        fb_name, platform, target_id,
                    )

                out_val = validate_output(analysis, evidence)
                pol_val = validate_policy(analysis)

                if out_val.is_valid and pol_val.is_valid:
                    structured_analysis_dict = analysis.model_dump()
                    base_row["structured_analysis"] = Jsonb(structured_analysis_dict)
                    base_row["llm_status"] = "generated"

                    rec_lines = []
                    if analysis.recommended_action:
                        rec_lines.append(f"Action: {analysis.recommended_action}")
                    if analysis.writer_recommendations:
                        rec_lines.append("• " + "\n• ".join(analysis.writer_recommendations[:2]))
                    if analysis.title_suggestions:
                        rec_lines.append(
                            "Suggested angles: "
                            + ", ".join(f'"{t}"' for t in analysis.title_suggestions[:2])
                        )
                    if analysis.publishing_timing:
                        rec_lines.append(f"Timing: {analysis.publishing_timing}")
                    ai_rec = (
                        "\n\n".join(rec_lines) if rec_lines else "Editorial recommendations generated."
                    )
                    base_row["ai_recommendation"] = ai_rec
                    # Update in-memory generated cache so sibling tasks can see it
                    existing_generated[target_id] = {
                        "ai_recommendation": ai_rec,
                        "structured_analysis": structured_analysis_dict,
                    }
                else:
                    logger.warning(
                        "[BulkScanner] LLM output validation failed for %s/%s: %s %s",
                        platform,
                        target_id,
                        out_val.failures,
                        pol_val.failures,
                    )
                    base_row["ai_recommendation"] = "Recommendation unavailable (validation failed)"
                    base_row["structured_analysis"] = None
                    base_row["llm_status"] = "failed_validation"

            except (ProviderUnavailableError, ProviderError, Exception) as exc:
                logger.warning(
                    "[BulkScanner] LLM generation failed for %s/%s: %s",
                    platform,
                    target_id,
                    exc,
                )
                base_row["ai_recommendation"] = "Recommendation unavailable"
                base_row["structured_analysis"] = None
                base_row["llm_status"] = "unavailable"

        return target_id, base_row

    # Fire all tasks and collect results; semaphore limits true concurrency
    llm_tasks = [_process_one(c) for c in llm_candidates]
    llm_results = await asyncio.gather(*llm_tasks, return_exceptions=False)

    # Tally counters and collect rows
    for tid, brow in llm_results:
        status = brow.get("llm_status")
        summary.recommendations_attempted += 1
        if status == "generated":
            summary.recommendations_generated += 1
        elif status == "failed_validation":
            summary.recommendations_failed_validation += 1
        else:
            summary.recommendations_unavailable += 1
        rows_to_upsert[tid] = brow

    # ── 6. Stage 4: Persist All Suggestions in Transaction ─────────────────
    try:
        async with pool.connection() as conn:
            for row in rows_to_upsert.values():
                try:
                    await _upsert_suggestion(conn, row)
                    summary.actionable += 1
                    summary.suggestions_updated += 1
                except Exception as exc:
                    logger.error(
                        "[BulkScanner] Failed to upsert suggestion for %s/%s: %s",
                        platform, row.get("content_id"), exc,
                    )
                    summary.errors += 1
            await conn.commit()
    except Exception as exc:
        logger.error("[BulkScanner] Transaction failure during scan upsert: %s", exc)
        summary.errors += 1

    # Remaining pending rows in DB (authoritative queue depth after this scan)
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT COUNT(*) FROM ai_suggestions
                    WHERE platform = %s AND llm_status = 'pending'
                    """,
                    (platform,),
                )
                row = await cur.fetchone()
                summary.recommendations_pending_remaining = int(row[0] or 0) if row else 0
    except Exception as exc:
        logger.warning("[BulkScanner] Could not count remaining pending rows: %s", exc)

    summary.finish()
    _duration_s = (
        (summary.finished_at - summary.started_at).total_seconds()
        if summary.finished_at
        else 0.0
    )
    logger.info(
        "[BulkScanner] %s scan complete: duration=%.1fs scanned=%d actionable=%d updated=%d "
        "cached=%d generated=%d unavailable=%d failed_validation=%d not_eligible=%d errors=%d "
        "pending_remaining=%d concurrency=%d provider=%s",
        platform,
        _duration_s,
        summary.scanned,
        summary.actionable,
        summary.suggestions_updated,
        summary.recommendations_cached,
        summary.recommendations_generated,
        summary.recommendations_unavailable,
        summary.recommendations_failed_validation,
        summary.recommendations_not_eligible,
        summary.errors,
        summary.recommendations_pending_remaining,
        _concurrency,
        summary.llm_provider,
    )
    return summary


async def scan_all_platforms(
    content_id: Optional[str] = None,
    max_items: Optional[int] = None,
    max_new_recommendations: Optional[int] = None,
    force_refresh: bool = False,
) -> List[ScanSummary]:
    """Run scan_platform() for all supported platforms and return summaries."""
    summaries = []
    for platform in ["youtube", "instagram", "facebook"]:
        summary = await scan_platform(
            platform,
            content_id=content_id,
            max_items=max_items,
            max_new_recommendations=max_new_recommendations,
            force_refresh=force_refresh,
        )
        summaries.append(summary)
    return summaries

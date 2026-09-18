"""
AI Suggestions API routes.

GET  /suggestions                       — all persisted actionable suggestions
GET  /suggestions/{platform}/{content_id} — single suggestion
GET  /notifications                     — recent actionable suggestions formatted as notifications
POST /scan                              — trigger a full bulk scan

All reads are from the ai_suggestions table (written by the bulk scanner).
POST /scan runs the scanner synchronously and returns the summary.
"""
from __future__ import annotations

import logging
from typing import LiteralString, Optional, cast

from fastapi import APIRouter, HTTPException, Query

from app.data_sources.postgres import _get_pool
from app.domain.models import GateClassification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


def _normalize_structured_analysis(value):  # noqa: ANN001
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    import json
    try:
        return json.loads(value)
    except Exception:
        return None


def _serialize_suggestion_row(d: dict) -> None:
    """Mutate suggestion dict in place for JSON API responses."""
    for k in ("scanned_at", "updated_at"):
        if d.get(k) is not None and hasattr(d[k], "isoformat"):
            d[k] = d[k].isoformat()
    if d.get("velocity_ratio") is not None:
        d["velocity_ratio"] = float(d["velocity_ratio"])
    if d.get("like_acceleration") is not None:
        d["like_acceleration"] = float(d["like_acceleration"])
    if d.get("coverage_hours") is not None:
        d["coverage_hours"] = float(d["coverage_hours"])
    if d.get("current_metric") is not None:
        d["current_metric"] = int(d["current_metric"])
    if d.get("baseline_metric") is not None:
        d["baseline_metric"] = float(d["baseline_metric"])
    if d.get("structured_analysis") is not None:
        d["structured_analysis"] = _normalize_structured_analysis(d["structured_analysis"])

_NOTIFICATION_CLASSIFICATIONS = {
    GateClassification.BOOMING_SURGE.value,
    GateClassification.SURGE_CANDIDATE.value,
    GateClassification.LOW_PERFORMING.value,
}


ACTIONABLE_CLASSIFICATIONS = [
    GateClassification.BOOMING_SURGE.value,
    GateClassification.SURGE_CANDIDATE.value,
    GateClassification.ELEVATED.value,
    GateClassification.LOW_PERFORMING.value,
]


async def _query_suggestions(
    platform: Optional[str],
    classification: Optional[str],
    limit: int,
    offset: int,
) -> tuple[list[dict], int, dict[str, int]]:
    pool = _get_pool()
    
    # Base filter for actionable records
    base_conditions = ["classification = ANY(%s)"]
    base_params: list = [ACTIONABLE_CLASSIFICATIONS]

    if platform:
        base_conditions.append("platform = %s")
        base_params.append(platform)

    base_where = f"WHERE {' AND '.join(base_conditions)}"

    # 1. Authoritative counts query across all 4 actionable categories in scope
    counts_sql = f"""
        WITH latest AS (
            SELECT DISTINCT ON (platform, content_id) classification
            FROM ai_suggestions
            {base_where}
            ORDER BY platform, content_id, analyzed_at DESC
        )
        SELECT classification, COUNT(*)
        FROM latest
        GROUP BY classification
    """

    # 2. Filtered items query
    item_conditions = list(base_conditions)
    item_params = list(base_params)

    if classification and classification != "all":
        item_conditions.append("classification = %s")
        item_params.append(classification)

    item_where = f"WHERE {' AND '.join(item_conditions)}"

    items_sql = f"""
        WITH latest AS (
            SELECT DISTINCT ON (platform, content_id)
                id, platform, content_id, account_key, title, classification,
                velocity_ratio, like_acceleration, current_metric, baseline_metric,
                evidence_reason as reason,
                ai_recommendation, structured_analysis, llm_status,
                report_period, coverage_hours,
                analyzed_at as scanned_at, is_low_performing, is_surge
            FROM ai_suggestions
            {item_where}
            ORDER BY platform, content_id, analyzed_at DESC
        )
        SELECT *
        FROM latest
        ORDER BY
          CASE classification
            WHEN 'BOOMING_SURGE' THEN 1
            WHEN 'SURGE_CANDIDATE' THEN 2
            WHEN 'LOW_PERFORMING' THEN 3
            WHEN 'ELEVATED' THEN 4
            ELSE 5
          END,
          CASE llm_status
            WHEN 'generated' THEN 1
            WHEN 'pending' THEN 2
            WHEN 'unavailable' THEN 3
            WHEN 'failed_validation' THEN 4
            WHEN 'not_eligible' THEN 5
            ELSE 6
          END,
          velocity_ratio DESC NULLS LAST,
          scanned_at DESC
        LIMIT %s OFFSET %s
    """

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Execute counts
            await cur.execute(cast(LiteralString, counts_sql), base_params)
            count_rows = await cur.fetchall()
            counts_map = {cls: 0 for cls in ACTIONABLE_CLASSIFICATIONS}
            for cls_name, c in count_rows:
                counts_map[cls_name] = int(c)

            # Determine total matching active filter
            if classification and classification != "all":
                total_in_filter = counts_map.get(classification, 0)
            else:
                total_in_filter = sum(counts_map.values())

            # Execute items query
            item_params_with_pagination = item_params + [limit, offset]
            await cur.execute(cast(LiteralString, items_sql), item_params_with_pagination)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = await cur.fetchall()

    from app.ml.readiness import get_cached_readiness

    items = [dict(zip(cols, row)) for row in rows]
    for item in items:
        item["xgboost_status"] = get_cached_readiness(item.get("platform"))
    return items, total_in_filter, counts_map


@router.get("")
async def get_suggestions(
    platform: Optional[str] = Query(None, description="Filter by platform (youtube|instagram|facebook)"),
    classification: Optional[str] = Query(None, description="Filter by classification"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Return all persisted AI suggestions with authoritative counts and single source of truth."""
    try:
        rows, total, counts_by_classification = await _query_suggestions(platform, classification, limit, offset)
        for row in rows:
            _serialize_suggestion_row(row)

        return {
            "ok": True,
            "suggestions": rows,
            "count": len(rows),
            "total": total,
            "counts_by_classification": counts_by_classification,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total,
            },
        }
    except Exception as exc:
        logger.error("[/suggestions] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch suggestions.")



@router.get("/notifications")
async def get_notifications(limit: int = Query(20, ge=1, le=100)):
    """
    Return recent actionable suggestions formatted as dashboard notifications.

    Only BOOMING_SURGE, SURGE_CANDIDATE, and LOW_PERFORMING are included.
    """
    try:
        pool = _get_pool()
        sql = """
            SELECT id, platform, content_id, classification, title, evidence_reason as reason, analyzed_at as scanned_at
            FROM ai_suggestions
            WHERE classification = ANY(%(classes)s)
            ORDER BY analyzed_at DESC
            LIMIT %(limit)s
        """
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(cast(LiteralString, sql), {
                    "classes": list(_NOTIFICATION_CLASSIFICATIONS),
                    "limit": limit,
                })
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = await cur.fetchall()

        notifications = []
        for row in rows:
            d = dict(zip(cols, row))
            cls = d["classification"]
            if cls == GateClassification.BOOMING_SURGE.value:
                ntype = "ai_surge"
                label = "🚀 Surge Detected"
            elif cls == GateClassification.SURGE_CANDIDATE.value:
                ntype = "ai_surge"
                label = "📈 Surge Candidate"
            else:
                ntype = "ai_low_performing"
                label = "⚠️ Low Performing"

            notifications.append({
                "id": f"ai-{d['platform']}-{d['content_id']}",
                "type": ntype,
                "platform": d["platform"],
                "contentId": d["content_id"],
                "title": d.get("title") or d["content_id"],
                "label": label,
                "message": d.get("reason", ""),
                "scanned_at": d["scanned_at"].isoformat() if d.get("scanned_at") else None,
            })

        return {"ok": True, "notifications": notifications, "count": len(notifications)}
    except Exception as exc:
        logger.error("[/suggestions/notifications] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch notifications.")


@router.get("/by-content/{content_id}")
async def get_suggestion_by_content(content_id: str):
    """
    Return the most recent AI suggestion for a content item across all platforms.
    Used for the clean detail route /dashboard/ai-suggestions/[contentId].
    """
    try:
        pool = _get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, platform, content_id, account_key, title, classification,
                           velocity_ratio, like_acceleration, current_metric, baseline_metric,
                           evidence_reason as reason,
                           ai_recommendation, structured_analysis, llm_status,
                           report_period, coverage_hours,
                           analyzed_at as scanned_at, is_low_performing, is_surge
                    FROM ai_suggestions
                    WHERE content_id = %s
                    ORDER BY analyzed_at DESC
                    LIMIT 1
                    """,
                    (content_id,),
                )
                row = await cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Suggestion not found.")
                cols = [d[0] for d in cur.description] if cur.description else []
                d = dict(zip(cols, row))
                from app.ml.readiness import get_cached_readiness
                d["xgboost_status"] = get_cached_readiness(d.get("platform"))
                _serialize_suggestion_row(d)
                return {"ok": True, "suggestion": d}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[/suggestions/by-content/%s] Error: %s", content_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch suggestion.")


@router.get("/{platform}/{content_id}")
async def get_suggestion(platform: str, content_id: str):
    """Return the AI suggestion for a single content item."""
    try:
        pool = _get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, platform, content_id, account_key, title, classification,
                           velocity_ratio, like_acceleration, current_metric, baseline_metric,
                           evidence_reason as reason,
                           ai_recommendation, structured_analysis, llm_status,
                           report_period, coverage_hours,
                           analyzed_at as scanned_at, is_low_performing, is_surge
                    FROM ai_suggestions
                    WHERE platform = %s AND content_id = %s
                    ORDER BY analyzed_at DESC
                    LIMIT 1
                    """,
                    (platform, content_id),
                )
                row = await cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Suggestion not found.")
                cols = [d[0] for d in cur.description] if cur.description else []
                d = dict(zip(cols, row))
                from app.ml.readiness import get_cached_readiness
                d["xgboost_status"] = get_cached_readiness(platform)
                _serialize_suggestion_row(d)
                return {"ok": True, "suggestion": d}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[/suggestions/%s/%s] Error: %s", platform, content_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch suggestion.")


@router.post("/scan")
async def trigger_scan(
    platforms: Optional[str] = Query(None, description="Comma-separated platforms to scan"),
    content_id: Optional[str] = Query(None, description="Optional content ID to scan"),
    max_items: Optional[int] = Query(None, description="Optional maximum content items to scan per platform"),
    batch_size: Optional[int] = Query(None, description="Ignored — all eligible items are processed per scan cycle. Accepted for backwards compatibility only."),
    force_refresh: bool = Query(False, description="Force re-generation of LLM recommendations"),
):
    """
    Trigger a synchronous bulk scan.

    Returns the scan summaries for each platform scanned.
    """
    from app.services.bulk_scanner import scan_all_platforms, scan_platform
    from app.services.email_reporter import send_scan_report

    try:
        if platforms:
            target_list = [p.strip().lower() for p in platforms.split(",") if p.strip()]
            summaries = []
            for p in target_list:
                s = await scan_platform(
                    p,
                    content_id=content_id,
                    max_items=max_items,
                    max_new_recommendations=batch_size,
                    force_refresh=force_refresh,
                )
                summaries.append(s)
        else:
            summaries = await scan_all_platforms(
                content_id=content_id,
                max_items=max_items,
                max_new_recommendations=batch_size,
                force_refresh=force_refresh,
            )

        summary_dicts = [
            {
                "platform": s.platform,
                "total_content_ids": s.total_content_ids,
                "scanned": s.scanned,
                "actionable": s.actionable,
                "skipped_insufficient": s.skipped_insufficient,
                "errors": s.errors,
                "suggestions_updated": s.suggestions_updated,
                "recommendations_attempted": s.recommendations_attempted,
                "recommendations_generated": s.recommendations_generated,
                "recommendations_cached": s.recommendations_cached,
                "recommendations_pending": s.recommendations_pending,
                "recommendations_pending_remaining": s.recommendations_pending_remaining,
                "recommendations_unavailable": s.recommendations_unavailable,
                "recommendations_failed_validation": s.recommendations_failed_validation,
                "recommendations_not_eligible": s.recommendations_not_eligible,
                "llm_provider": s.llm_provider,
                "xgboost_status": s.xgboost_status,
                "started_at": s.started_at.isoformat(),
                "finished_at": s.finished_at.isoformat() if s.finished_at else None,
            }
            for s in summaries
        ]

        # Send optional email report if configured
        send_scan_report(summary_dicts)

        return {
            "ok": True,
            "message": f"Scan completed across {len(summaries)} platform(s).",
            "summaries": summary_dicts,
        }
    except Exception as exc:
        logger.error("[/suggestions/scan] Error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Scan failed: {exc}")


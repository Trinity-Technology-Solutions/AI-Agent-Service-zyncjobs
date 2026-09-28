"""
AI Suggestions API routes for Performance Insights.

GET  /suggestions                       — all persisted authoritative performance insights
GET  /suggestions/{platform}/{content_id} — single suggestion
GET  /suggestions/by-content/{content_id} — suggestion by content ID across platforms
GET  /suggestions/notifications          — recent actionable insights formatted as notifications
POST /suggestions/scan                   — trigger scan for authoritative dashboard candidates

All data is strictly derived from the dashboard performance engine and verified evidence.
The AI service does not calculate, classify, gate, rank, or predict performance.
"""
from __future__ import annotations

import asyncio as _asyncio
import logging
from typing import Any, LiteralString, Optional, cast

from fastapi import APIRouter, HTTPException, Query, Request

from app.data_sources.postgres import _get_pool
from app.domain.models import PerformanceClassification
from app.services.bulk_scanner import scan_with_candidates

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/suggestions", tags=["suggestions"])

_SCAN_LOCK = _asyncio.Lock()

ACTIONABLE_CLASSIFICATIONS = [
    PerformanceClassification.HIGH_PERFORMING.value,
    PerformanceClassification.LOW_PERFORMING.value,
]

_NOTIFICATION_CLASSIFICATIONS = set(ACTIONABLE_CLASSIFICATIONS)


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
    for k in ("scanned_at", "analyzed_at", "updated_at"):
        if d.get(k) is not None and hasattr(d[k], "isoformat"):
            d[k] = d[k].isoformat()
    if d.get("current_metric") is not None:
        d["current_metric"] = int(d["current_metric"])
    if d.get("baseline_metric") is not None:
        d["baseline_metric"] = float(d["baseline_metric"])
    if d.get("likes") is not None:
        d["likes"] = int(d["likes"])
    if d.get("comments") is not None:
        d["comments"] = int(d["comments"])
    if d.get("structured_analysis") is not None:
        d["structured_analysis"] = _normalize_structured_analysis(d["structured_analysis"])


async def _query_suggestions(
    platform: Optional[str],
    classification: Optional[str],
    period: Optional[str],
    limit: int,
    offset: int,
) -> tuple[list[dict], int, dict[str, int]]:
    pool = _get_pool()

    base_conditions = ["classification = ANY(%s)"]
    base_params: list = [ACTIONABLE_CLASSIFICATIONS]

    if platform and platform.lower() != "all":
        base_conditions.append("platform = %s")
        base_params.append(platform.lower().strip())

    if period and period.lower() != "all":
        base_conditions.append("report_period = %s")
        base_params.append(period.lower().strip())

    base_where = f"WHERE {' AND '.join(base_conditions)}"

    # 1. Authoritative counts query across active scope
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

    if classification and classification.upper() != "ALL":
        item_conditions.append("classification = %s")
        item_params.append(classification.upper().strip())

    item_where = f"WHERE {' AND '.join(item_conditions)}"

    items_sql = f"""
        WITH latest AS (
            SELECT DISTINCT ON (platform, content_id)
                id, platform, content_id, account_key, title,
                content_type, canonical_url, metric_name,
                classification, current_metric, baseline_metric,
                likes, comments, peer_explanation,
                evidence_reason as reason,
                ai_recommendation, structured_analysis, llm_status,
                report_period,
                analyzed_at as scanned_at, is_low_performing
            FROM ai_suggestions
            {item_where}
            ORDER BY platform, content_id, analyzed_at DESC
        )
        SELECT * FROM latest
        ORDER BY
          CASE classification
            WHEN 'HIGH_PERFORMING' THEN 1
            WHEN 'LOW_PERFORMING' THEN 2
            ELSE 3
          END,
          CASE
            WHEN classification = 'HIGH_PERFORMING' THEN current_metric
            ELSE NULL
          END DESC NULLS LAST,
          CASE
            WHEN classification = 'LOW_PERFORMING' THEN current_metric
            ELSE NULL
          END ASC NULLS LAST,
          scanned_at DESC
        LIMIT %s OFFSET %s
    """

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(cast(LiteralString, counts_sql), base_params)
            count_rows = await cur.fetchall()
            counts_map = {cls: 0 for cls in ACTIONABLE_CLASSIFICATIONS}
            for cls_name, c in count_rows:
                counts_map[cls_name] = int(c)

            if classification and classification.upper() != "ALL":
                total_in_filter = counts_map.get(classification.upper(), 0)
            else:
                total_in_filter = sum(counts_map.values())

            item_params_with_pagination = item_params + [limit, offset]
            await cur.execute(cast(LiteralString, items_sql), item_params_with_pagination)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = await cur.fetchall()

    items = [dict(zip(cols, row)) for row in rows]
    return items, total_in_filter, counts_map


@router.get("")
async def get_suggestions(
    platform: Optional[str] = Query(None, description="Filter by platform (youtube|instagram|facebook)"),
    classification: Optional[str] = Query(None, description="Filter by classification (HIGH_PERFORMING|LOW_PERFORMING)"),
    period: Optional[str] = Query(None, description="Filter by report period (7d|30d|90d)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Return all persisted AI performance insights with authoritative counts."""
    try:
        rows, total, counts_by_classification = await _query_suggestions(
            platform, classification, period, limit, offset
        )
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
    """Return recent actionable suggestions formatted as dashboard notifications."""
    try:
        pool = _get_pool()
        sql = """
            SELECT id, platform, content_id, classification, title, content_type,
                   canonical_url, evidence_reason as reason, analyzed_at as scanned_at
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
            if cls == PerformanceClassification.HIGH_PERFORMING.value:
                ntype = "ai_high_performing"
                label = "🏆 High Performing"
            else:
                ntype = "ai_low_performing"
                label = "⚠️ Low Performing"

            notifications.append({
                "id": f"ai-{d['platform']}-{d['content_id']}",
                "type": ntype,
                "platform": d["platform"],
                "contentId": d["content_id"],
                "title": d.get("title") or d["content_id"],
                "contentType": d.get("content_type", "Video"),
                "canonicalUrl": d.get("canonical_url"),
                "label": label,
                "message": d.get("reason", ""),
                "scanned_at": d["scanned_at"].isoformat() if d.get("scanned_at") else None,
            })

        return {"ok": True, "notifications": notifications, "count": len(notifications)}
    except Exception as exc:
        logger.error("[/suggestions/notifications] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch notifications.")


@router.get("/by-content/{content_id}")
async def get_suggestion_by_content(
    content_id: str,
    period: Optional[str] = Query(None, description="Optional period filter (7d|30d|90d)"),
):
    """Return the most recent AI suggestion for a content item across all platforms, optionally scoped to period."""
    try:
        pool = _get_pool()
        conditions = ["content_id = %s", "classification = ANY(%s)"]
        params: list[Any] = [content_id, ACTIONABLE_CLASSIFICATIONS]
        if period:
            conditions.append("LOWER(report_period) = %s")
            params.append(period.lower().strip())
        where_clause = " AND ".join(conditions)

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"""
                    SELECT id, platform, content_id, account_key, title,
                           content_type, canonical_url, metric_name,
                           classification, current_metric, baseline_metric,
                           likes, comments, peer_explanation,
                           evidence_reason as reason,
                           ai_recommendation, structured_analysis, llm_status,
                           report_period,
                           analyzed_at as scanned_at, is_low_performing
                    FROM ai_suggestions
                    WHERE {where_clause}
                    ORDER BY analyzed_at DESC
                    LIMIT 1
                    """,
                    params,
                )
                row = await cur.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Suggestion not found.")
                cols = [d[0] for d in cur.description] if cur.description else []
                d = dict(zip(cols, row))
                _serialize_suggestion_row(d)
                return {"ok": True, "suggestion": d}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[/suggestions/by-content/%s] Error: %s", content_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch suggestion.")


@router.get("/{platform}/{content_id}")
async def get_suggestion(platform: str, content_id: str):
    """Return the AI suggestion for a single content item on a specific platform."""
    try:
        pool = _get_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, platform, content_id, account_key, title,
                           content_type, canonical_url, metric_name,
                           classification, current_metric, baseline_metric,
                           likes, comments, peer_explanation,
                           evidence_reason as reason,
                           ai_recommendation, structured_analysis, llm_status,
                           report_period,
                           analyzed_at as scanned_at, is_low_performing
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
                _serialize_suggestion_row(d)
                return {"ok": True, "suggestion": d}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[/suggestions/%s/%s] Error: %s", platform, content_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch suggestion.")


@router.post("/scan")
async def trigger_scan(
    request: Request,
    platforms: Optional[str] = Query(None, description="Comma-separated platforms to scan"),
    content_id: Optional[str] = Query(None, description="Optional content ID to scan"),
    period: Optional[str] = Query("30d", description="Analysis period (7d|30d|90d)"),
    force_refresh: bool = Query(False, description="Force re-generation of LLM recommendations"),
):
    """
    Trigger an AI Performance Insights scan for authoritative dashboard candidates.

    Expects a JSON body containing:
      - candidates: list of authoritative HIGH_PERFORMING and LOW_PERFORMING items
                    from the dashboard performance engine.
      - period, platforms, content_id, force_refresh: optional override fields.

    ARCHITECTURAL RULE:
    The AI service does NOT classify or gate performance.
    If no candidates are supplied, an explicit error state is returned.
    There is no fallback to legacy velocity or ML scanning.
    """
    if _SCAN_LOCK.locked():
        return {
            "ok": False,
            "error": "A scan is already in progress. Please wait for it to complete before triggering another.",
            "error_type": "scan_in_progress",
        }

    body_candidates: list = []
    body_period: str = period or "30d"
    body_force_refresh: bool = force_refresh
    body_platforms: Optional[str] = platforms
    body_content_id: Optional[str] = content_id

    try:
        body = await request.json()
        if isinstance(body, dict):
            raw_candidates = body.get("candidates")
            if isinstance(raw_candidates, list):
                body_candidates = raw_candidates
            if "period" in body and body["period"]:
                body_period = str(body["period"]).lower()
            if "force_refresh" in body:
                body_force_refresh = bool(body["force_refresh"])
            if "platforms" in body and body["platforms"]:
                body_platforms = body_platforms or str(body["platforms"])
            if "content_id" in body and body["content_id"]:
                body_content_id = body_content_id or str(body["content_id"])
    except Exception:
        pass  # Query params are used if body is not JSON

    # Filter candidates to requested platforms if specified
    if body_platforms and body_platforms != "all":
        req_plats = {p.strip().lower() for p in body_platforms.split(",") if p.strip()}
        filtered = [c for c in body_candidates if (c.get("platform") or "").lower() in req_plats]
    else:
        filtered = body_candidates

    # Filter to requested content_id if specified
    if body_content_id:
        filtered = [c for c in filtered if str(c.get("content_id") or "") == body_content_id]

    # Explicit error state when no authoritative candidates are available
    if not filtered:
        return {
            "ok": False,
            "error": f"No authoritative performance candidates provided for {body_platforms or 'all'} in {body_period.upper()} period. Ensure content exists in the dashboard performance engine.",
            "error_type": "no_candidates",
            "period": body_period,
            "platforms": body_platforms,
        }

    logger.info(
        "[/suggestions/scan] Processing %d authoritative candidates (period=%s, force_refresh=%s)",
        len(filtered), body_period, body_force_refresh,
    )

    async with _SCAN_LOCK:
        try:
            summaries = await scan_with_candidates(
                filtered, force_refresh=body_force_refresh, period=body_period
            )
            return {
                "ok": True,
                "period": body_period,
                "candidates_processed": len(filtered),
                "summaries": [s.__dict__ for s in summaries],
            }
        except Exception as exc:
            logger.error("[/suggestions/scan] Scan failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Candidate scan failed: {exc}")

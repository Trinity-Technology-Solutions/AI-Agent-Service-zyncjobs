"""
Reports API routes.

GET /reports?period=hourly|daily|monthly|yearly&platform=...
  Returns aggregated statistics from real history observation timestamps.
  Each bucket is one unique (platform scope, period) time window with aggregated
  observation counts, distinct content items, platform primary metrics, and
  current surge/low-performing signal counts joined by content_id (not scan time).
  When history is insufficient for the requested period, returns INSUFFICIENT_COVERAGE.

GET /reports/intelligence?platform=...&days=...
  Performance Intelligence: deduplicated daily observations for executive charting.
  Proxies to /intelligence/performance — single authoritative source of truth.

GET /reports/publishing-time?platform=...
  Observed publishing-time analysis (NEVER predicted).
  Proxies to /intelligence/publishing — single authoritative source of truth.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from app.data_sources.postgres import _PLATFORM_CFG, _get_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["reports"])

_PERIOD_TRUNC = {
    "hourly": "hour",
    "daily": "day",
    "monthly": "month",
    "yearly": "year",
}

_REQUIRED_HOURS = {
    "hourly": 2.0,
    "daily": 24.0,
    "monthly": 720.0,
    "yearly": 8760.0,
}


def _primary_metric_label(platforms: list[str]) -> str:
    if len(platforms) == 1:
        p = platforms[0]
        metric = _PLATFORM_CFG[p]["primary_metric"]
        return "views" if metric == "views" else "reach"
    return "primary_metric"


def _build_history_union(platforms: list[str]) -> str:
    parts: list[str] = []
    for p in platforms:
        cfg = _PLATFORM_CFG[p]
        metric = cfg["primary_metric"]
        id_col = cfg["id_col"]
        table = cfg["history_table"]
        parts.append(
            f"SELECT '{p}'::text AS platform, {id_col}::text AS content_id, "
            f"collected_at, {metric}::bigint AS primary_metric, "
            f"likes::bigint AS likes, comments::bigint AS comments "
            f"FROM {table}"
        )
    return " UNION ALL ".join(parts)


@router.get("")
async def get_reports(
    period: Literal["hourly", "daily", "monthly", "yearly"] = Query("daily"),
    platform: Optional[str] = Query(None, description="Optional platform filter"),
    limit: int = Query(30, ge=1, le=365),
):
    trunc = _PERIOD_TRUNC.get(period, "day")
    req_hours = _REQUIRED_HOURS.get(period, 24.0)

    try:
        pool = _get_pool()

        platforms_to_check = (
            [platform]
            if platform and platform in _PLATFORM_CFG
            else ["youtube", "instagram", "facebook"]
        )
        metric_label = _primary_metric_label(platforms_to_check)

        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                earliest_dates = []
                latest_dates = []
                for p in platforms_to_check:
                    tbl = _PLATFORM_CFG[p]["history_table"]
                    try:
                        await cur.execute(
                            f"SELECT MIN(collected_at), MAX(collected_at) FROM {tbl}"
                        )
                        row = await cur.fetchone()
                        if row and row[0] and row[1]:
                            earliest_dates.append(row[0])
                            latest_dates.append(row[1])
                    except Exception as e:
                        logger.warning("[/reports] Could not query span for %s: %s", p, e)

                span_hours = 0.0
                if earliest_dates and latest_dates:
                    span_hours = (
                        max(latest_dates) - min(earliest_dates)
                    ).total_seconds() / 3600.0

                avail_hours = round(span_hours, 1)
                avail_days = round(span_hours / 24.0, 1)
                req_days = round(req_hours / 24.0, 1)

                if span_hours < req_hours:
                    coverage_status = "INSUFFICIENT_COVERAGE"
                    reason = (
                        f"Historical data covers only {avail_days} days ({avail_hours} hours). "
                        f"A {period} report requires at least {req_days} days "
                        f"({req_hours:.0f} hours) of continuous history."
                    )
                else:
                    coverage_status = "SUFFICIENT"
                    reason = (
                        f"Available history ({avail_days} days) satisfies the required "
                        f"{req_days} days ({req_hours:.0f} hours) for {period} reporting."
                    )

                history_union = _build_history_union(platforms_to_check)

                suggestion_cond = "WHERE 1=1"
                suggestion_params: list = []
                if platform:
                    suggestion_cond += " AND platform = %s"
                    suggestion_params.append(platform)

                sql_buckets = f"""
                    WITH history_rows AS (
                        {history_union}
                    ),
                    history_buckets AS (
                        SELECT
                            DATE_TRUNC('{trunc}', collected_at) AS period_start,
                            COUNT(*) AS total_observations,
                            COUNT(DISTINCT (platform, content_id)) AS content_items_observed,
                            COALESCE(SUM(primary_metric), 0) AS total_primary_metric,
                            COALESCE(SUM(likes), 0) AS total_likes,
                            COALESCE(SUM(comments), 0) AS total_comments
                        FROM history_rows
                        GROUP BY period_start
                    ),
                    signal_join AS (
                        SELECT
                            DATE_TRUNC('{trunc}', h.collected_at) AS period_start,
                            COUNT(DISTINCT (h.platform, h.content_id)) FILTER (
                                WHERE s.classification = 'BOOMING_SURGE'
                            ) AS booming_surge,
                            COUNT(DISTINCT (h.platform, h.content_id)) FILTER (
                                WHERE s.classification = 'SURGE_CANDIDATE'
                            ) AS surge_candidate,
                            COUNT(DISTINCT (h.platform, h.content_id)) FILTER (
                                WHERE s.is_low_performing = true
                            ) AS low_performing
                        FROM history_rows h
                        LEFT JOIN ai_suggestions s
                            ON s.platform = h.platform AND s.content_id = h.content_id
                        GROUP BY period_start
                    )
                    SELECT
                        hb.period_start,
                        hb.total_observations,
                        hb.content_items_observed,
                        hb.total_primary_metric,
                        hb.total_likes,
                        hb.total_comments,
                        COALESCE(sj.booming_surge, 0) AS booming_surge,
                        COALESCE(sj.surge_candidate, 0) AS surge_candidate,
                        COALESCE(sj.low_performing, 0) AS low_performing
                    FROM history_buckets hb
                    LEFT JOIN signal_join sj ON hb.period_start = sj.period_start
                    ORDER BY hb.period_start DESC
                    LIMIT %s
                """
                await cur.execute(sql_buckets, [limit])
                b_cols = [d[0] for d in cur.description]
                b_rows = await cur.fetchall()

                buckets_raw = [dict(zip(b_cols, b)) for b in b_rows]

                # Compute period-over-period metric change (chronological)
                asc = sorted(
                    buckets_raw,
                    key=lambda x: x["period_start"] if x.get("period_start") else "",
                )
                prev_metric: Optional[int] = None
                for row in asc:
                    curr = int(row.get("total_primary_metric") or 0)
                    if prev_metric is not None and prev_metric > 0:
                        row["metric_change_pct"] = round(
                            ((curr - prev_metric) / prev_metric) * 100.0, 1
                        )
                    else:
                        row["metric_change_pct"] = None
                    prev_metric = curr

                buckets = []
                for d in sorted(
                    buckets_raw,
                    key=lambda x: x["period_start"] if x.get("period_start") else "",
                    reverse=True,
                ):
                    surge_signals = int(d.get("booming_surge") or 0) + int(
                        d.get("surge_candidate") or 0
                    )
                    buckets.append({
                        "period_start": d["period_start"].isoformat()
                        if d.get("period_start")
                        else None,
                        "period": period,
                        "total_observations": int(d["total_observations"]),
                        "content_items_observed": int(d["content_items_observed"]),
                        "total_primary_metric": int(d["total_primary_metric"]),
                        "primary_metric_label": metric_label,
                        "total_likes": int(d["total_likes"]),
                        "total_comments": int(d["total_comments"]),
                        "metric_change_pct": d.get("metric_change_pct"),
                        "surge_signals": surge_signals,
                        "booming_surge": int(d["booming_surge"]),
                        "surge_candidate": int(d["surge_candidate"]),
                        "low_performing": int(d["low_performing"]),
                        # Legacy field kept for clients; equals distinct content with activity in bucket
                        "total_suggestions": int(d["content_items_observed"]),
                    })

                p_cond = ""
                p_params: list = []
                if platform:
                    p_cond = "WHERE platform = %s"
                    p_params.append(platform)

                await cur.execute(
                    f"""
                    SELECT
                        COUNT(DISTINCT content_id) AS tracked_content,
                        COUNT(*) AS total_actionable,
                        COUNT(*) FILTER (WHERE is_surge = true) AS total_surges,
                        COUNT(*) FILTER (WHERE is_low_performing = true) AS total_low_performing
                    FROM ai_suggestions
                    {p_cond}
                    """,
                    p_params,
                )
                sum_row = await cur.fetchone()
                summary_data = {
                    "tracked_content": int(sum_row[0] or 0),
                    "total_actionable": int(sum_row[1] or 0),
                    "total_surges": int(sum_row[2] or 0),
                    "total_low_performing": int(sum_row[3] or 0),
                }

                item_cols = [
                    "id", "platform", "content_id", "title", "classification",
                    "velocity_ratio", "like_acceleration", "current_metric",
                    "baseline_metric", "coverage_hours", "evidence_reason",
                    "ai_recommendation", "analyzed_at",
                ]
                item_sql_cols = ", ".join(item_cols)
                plat_filter = "AND platform = %s" if platform else ""

                await cur.execute(
                    f"""
                    SELECT {item_sql_cols}
                    FROM ai_suggestions
                    WHERE is_surge = true {plat_filter}
                    ORDER BY velocity_ratio DESC NULLS LAST
                    LIMIT 15
                    """,
                    [platform] if platform else [],
                )
                surge_items = _serialize_items(item_cols, await cur.fetchall())

                await cur.execute(
                    f"""
                    SELECT {item_sql_cols}
                    FROM ai_suggestions
                    WHERE is_low_performing = true {plat_filter}
                    ORDER BY velocity_ratio ASC NULLS LAST
                    LIMIT 15
                    """,
                    [platform] if platform else [],
                )
                low_items = _serialize_items(item_cols, await cur.fetchall())

        total_obs = sum(b["total_observations"] for b in buckets)
        total_content_in_buckets = sum(b["content_items_observed"] for b in buckets)
        total_surge_signals = sum(b["surge_signals"] for b in buckets)

        movement_note = None
        if len(buckets) >= 2 and buckets[0].get("metric_change_pct") is not None:
            pct = buckets[0]["metric_change_pct"]
            direction = "increased" if pct > 0 else "decreased" if pct < 0 else "unchanged"
            movement_note = (
                f"Latest {period} bucket {metric_label} {direction} "
                f"{abs(pct):.1f}% vs the previous bucket ({pct:+.1f}%)."
            )

        executive_summary = {
            "reporting_period": period,
            "coverage_status": coverage_status,
            "available_days": avail_days,
            "required_days": req_days,
            "bucket_count": len(buckets),
            "total_observations": total_obs,
            "content_items_observed": total_content_in_buckets,
            "surge_signal_observations": total_surge_signals,
            "tracked_content": summary_data["tracked_content"],
            "notable_movement": movement_note,
        }

        return {
            "ok": True,
            "period": period,
            "coverage_status": coverage_status,
            "coverage": {
                "available_hours": avail_hours,
                "available_days": avail_days,
                "required_hours": req_hours,
                "required_days": req_days,
                "reason": reason,
            },
            "executive_summary": executive_summary,
            "performance_summary": summary_data,
            "detected_surges": surge_items,
            "low_performing": low_items,
            "buckets": buckets,
        }

    except Exception as exc:
        logger.error("[/reports] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate report.")


def _serialize_items(cols: list[str], rows: list) -> list[dict]:
    items = []
    for r in rows:
        item = dict(zip(cols, r))
        if item.get("analyzed_at"):
            item["analyzed_at"] = item["analyzed_at"].isoformat()
            item["scanned_at"] = item["analyzed_at"]
        if item.get("velocity_ratio") is not None:
            item["velocity_ratio"] = float(item["velocity_ratio"])
        if item.get("like_acceleration") is not None:
            item["like_acceleration"] = float(item["like_acceleration"])
        if item.get("baseline_metric") is not None:
            item["baseline_metric"] = float(item["baseline_metric"])
        if item.get("coverage_hours") is not None:
            item["coverage_hours"] = float(item["coverage_hours"])
        if item.get("current_metric") is not None:
            item["current_metric"] = int(item["current_metric"])
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# GET /reports/intelligence
# Thin proxy → /intelligence/performance
# Keeps a stable /reports/ URL surface for the dashboard backend.
# ---------------------------------------------------------------------------

@router.get("/intelligence")
async def get_performance_intelligence_report(
    platform: Optional[str] = Query(None, description="Filter by platform"),
    days: int = Query(30, ge=7, le=90, description="Lookback window in days"),
):
    """
    Performance Intelligence visualizations derived from deduplicated real observations.

    Delegates to the canonical /intelligence/performance endpoint.
    All data is from verified *_history_ai records — no fabrication.
    """
    from app.api.routes.intelligence import get_performance_intelligence
    return await get_performance_intelligence(platform=platform, days=days)


# ---------------------------------------------------------------------------
# GET /reports/publishing-time
# Thin proxy → /intelligence/publishing
# ---------------------------------------------------------------------------

@router.get("/publishing-time")
async def get_publishing_time_report(
    platform: Optional[str] = Query(None, description="Filter by platform"),
    min_observations: int = Query(
        20,
        ge=5,
        description="Minimum observations required before reporting a publishing window",
    ),
):
    """
    Observed publishing-time analysis.

    Returns historically observed peak engagement windows derived ONLY from
    verified collected_at timestamps.  Returns 'insufficient_evidence' when
    there is not enough data rather than fabricating a result.

    Delegates to the canonical /intelligence/publishing endpoint.
    """
    from app.api.routes.intelligence import get_publishing_intelligence
    return await get_publishing_intelligence(
        platform=platform, min_observations=min_observations
    )

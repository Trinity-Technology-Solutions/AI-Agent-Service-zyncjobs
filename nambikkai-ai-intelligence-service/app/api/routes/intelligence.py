"""
Intelligence API routes.

GET  /intelligence/performance   — Performance chart data (deduplicated daily observations)
GET  /intelligence/publishing    — Observed publishing time analysis per platform
POST /intelligence/chat          — Grounded chatbot: answers factual questions from verified DB

All data is derived strictly from verified records in ai_suggestions and *_history_ai tables.
No fabrication, interpolation, or prediction presented as observed fact.
"""
from __future__ import annotations

import logging
import re
from typing import LiteralString, Optional, cast

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.data_sources.postgres import _PLATFORM_CFG, _get_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intelligence", tags=["intelligence"])

VALID_PLATFORMS = list(_PLATFORM_CFG.keys())  # ["youtube", "instagram", "facebook"]


# ---------------------------------------------------------------------------
# Helper: build a history UNION across requested platforms
# ---------------------------------------------------------------------------

def _build_history_union(platforms: list[str]) -> str:
    parts: list[str] = []
    for p in platforms:
        cfg = _PLATFORM_CFG[p]
        metric = cfg["primary_metric"]
        id_col = cfg["id_col"]
        table = cfg["history_table"]
        account_col = cfg["account_col"]
        parts.append(
            f"SELECT '{p}'::text AS platform, {id_col}::text AS content_id, "
            f"{account_col}::text AS account_key, "
            f"collected_at, {metric}::bigint AS primary_metric, "
            f"likes::bigint AS likes, comments::bigint AS comments "
            f"FROM {table}"
        )
    return " UNION ALL ".join(parts)


# ---------------------------------------------------------------------------
# GET /intelligence/performance
# ---------------------------------------------------------------------------

@router.get("/performance")
async def get_performance_intelligence(
    platform: Optional[str] = Query(None, description="Filter by platform"),
    days: int = Query(30, ge=7, le=90, description="Lookback window in days"),
):
    """
    Return deduplicated daily performance observations for charting.

    Uses DATE_TRUNC('day', collected_at) to produce one row per (platform, content_id, day),
    avoiding inflation from repeated ingestion of the same data point.

    Returns:
    - daily_series: [{date, total_primary_metric, total_likes, total_comments,
                       content_items_observed, surge_signals, low_performing_signals}]
    - platform_breakdown: [{platform, total_primary_metric, content_items, obs_days}]
    - top_performers: top 5 content items by total observed primary metric
    - coverage: {observed_days, first_observation, last_observation, total_observations}
    """
    try:
        platforms = [platform] if platform and platform in VALID_PLATFORMS else VALID_PLATFORMS
        union = _build_history_union(platforms)
        pool = _get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                # 1. Deduplicated daily series — one row per (platform, content_id, day)
                await cur.execute(
                    cast(
                        LiteralString,
                        f"""
                    WITH deduped AS (
                        SELECT
                            platform,
                            content_id,
                            DATE_TRUNC('day', collected_at) AS obs_day,
                            MAX(primary_metric) AS metric,
                            MAX(likes) AS likes,
                            MAX(comments) AS comments
                        FROM ({union}) h
                        WHERE collected_at >= NOW() - INTERVAL '{days} days'
                        GROUP BY platform, content_id, obs_day
                    ),
                    daily AS (
                        SELECT
                            obs_day,
                            SUM(metric) AS total_primary_metric,
                            SUM(likes) AS total_likes,
                            SUM(comments) AS total_comments,
                            COUNT(DISTINCT content_id) AS content_items_observed
                        FROM deduped
                        GROUP BY obs_day
                        ORDER BY obs_day
                    )
                    SELECT
                        daily.*,
                        COALESCE(surge_counts.surge_signals, 0) AS surge_signals,
                        COALESCE(lp_counts.low_signals, 0) AS low_performing_signals
                    FROM daily
                    LEFT JOIN (
                        SELECT DATE_TRUNC('day', analyzed_at) AS d, COUNT(*) AS surge_signals
                        FROM ai_suggestions
                        WHERE classification IN ('BOOMING_SURGE','SURGE_CANDIDATE')
                          AND analyzed_at >= NOW() - INTERVAL '{days} days'
                        GROUP BY d
                    ) surge_counts ON surge_counts.d = daily.obs_day
                    LEFT JOIN (
                        SELECT DATE_TRUNC('day', analyzed_at) AS d, COUNT(*) AS low_signals
                        FROM ai_suggestions
                        WHERE classification = 'LOW_PERFORMING'
                          AND analyzed_at >= NOW() - INTERVAL '{days} days'
                        GROUP BY d
                    ) lp_counts ON lp_counts.d = daily.obs_day
                    ORDER BY daily.obs_day
                    """,
                    )
                )
                daily_rows = await cur.fetchall()
                daily_cols = [d[0] for d in cur.description] if cur.description else []

                # 2. Platform breakdown
                await cur.execute(
                    cast(
                        LiteralString,
                        f"""
                    WITH deduped AS (
                        SELECT
                            platform,
                            content_id,
                            DATE_TRUNC('day', collected_at) AS obs_day,
                            MAX(primary_metric) AS metric
                        FROM ({union}) h
                        WHERE collected_at >= NOW() - INTERVAL '{days} days'
                        GROUP BY platform, content_id, obs_day
                    )
                    SELECT
                        platform,
                        SUM(metric) AS total_primary_metric,
                        COUNT(DISTINCT content_id) AS content_items,
                        COUNT(DISTINCT obs_day) AS obs_days
                    FROM deduped
                    GROUP BY platform
                    ORDER BY total_primary_metric DESC
                    """,
                    )
                )
                platform_rows = await cur.fetchall()
                platform_cols = [d[0] for d in cur.description] if cur.description else []

                # 3. Top performers from ai_suggestions (verified classifications)
                platform_filter = "AND platform = ANY(%s)" if platform else ""
                platform_filter_param = [platforms] if platform else []
                await cur.execute(
                    cast(
                        LiteralString,
                        f"""
                    WITH latest AS (
                        SELECT DISTINCT ON (platform, content_id)
                            platform, content_id, title, classification,
                            current_metric, velocity_ratio, analyzed_at
                        FROM ai_suggestions
                        WHERE classification IN ('BOOMING_SURGE','SURGE_CANDIDATE','ELEVATED','LOW_PERFORMING')
                        {platform_filter}
                        ORDER BY platform, content_id, analyzed_at DESC
                    )
                    SELECT platform, content_id, title, classification, current_metric, velocity_ratio
                    FROM latest
                    ORDER BY velocity_ratio DESC NULLS LAST, current_metric DESC NULLS LAST
                    LIMIT 5
                    """,
                    ),
                    platform_filter_param if platform else [],
                )
                top_rows = await cur.fetchall()
                top_cols = [d[0] for d in cur.description] if cur.description else []

                # 4. Coverage metadata
                await cur.execute(
                    cast(
                        LiteralString,
                        f"""
                    SELECT
                        COUNT(DISTINCT DATE_TRUNC('day', collected_at)) AS observed_days,
                        MIN(collected_at) AS first_observation,
                        MAX(collected_at) AS last_observation,
                        COUNT(*) AS total_observations
                    FROM ({union}) h
                    WHERE collected_at >= NOW() - INTERVAL '{days} days'
                    """,
                    )
                )
                cov_row = await cur.fetchone()

        # Serialize
        daily_series = []
        for row in daily_rows:
            d = dict(zip(daily_cols, row))
            d["date"] = d["obs_day"].isoformat() if d.get("obs_day") else None
            d.pop("obs_day", None)
            for k in ("total_primary_metric", "total_likes", "total_comments"):
                d[k] = int(d[k]) if d.get(k) is not None else 0
            d["content_items_observed"] = int(d.get("content_items_observed") or 0)
            d["surge_signals"] = int(d.get("surge_signals") or 0)
            d["low_performing_signals"] = int(d.get("low_performing_signals") or 0)
            daily_series.append(d)

        platform_breakdown = []
        for row in platform_rows:
            d = dict(zip(platform_cols, row))
            d["total_primary_metric"] = int(d.get("total_primary_metric") or 0)
            d["content_items"] = int(d.get("content_items") or 0)
            d["obs_days"] = int(d.get("obs_days") or 0)
            platform_breakdown.append(d)

        top_performers = []
        for row in top_rows:
            d = dict(zip(top_cols, row))
            if d.get("velocity_ratio") is not None:
                d["velocity_ratio"] = float(d["velocity_ratio"])
            if d.get("current_metric") is not None:
                d["current_metric"] = int(d["current_metric"])
            top_performers.append(d)

        coverage = {
            "observed_days": int(cov_row[0] or 0) if cov_row else 0,
            "first_observation": cov_row[1].isoformat() if cov_row and cov_row[1] else None,
            "last_observation": cov_row[2].isoformat() if cov_row and cov_row[2] else None,
            "total_observations": int(cov_row[3] or 0) if cov_row else 0,
        }

        from app.ml.readiness import get_xgboost_status_detail
        xgb_platform = platforms[0] if len(platforms) == 1 else None
        xgboost_status = get_xgboost_status_detail(xgb_platform)

        return {
            "ok": True,
            "days_requested": days,
            "platforms": platforms,
            "daily_series": daily_series,
            "platform_breakdown": platform_breakdown,
            "top_performers": top_performers,
            "coverage": coverage,
            "xgboost_status": xgboost_status,
        }

    except Exception as exc:
        logger.error("[/intelligence/performance] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch performance intelligence.")


# ---------------------------------------------------------------------------
# GET /intelligence/publishing
# ---------------------------------------------------------------------------

@router.get("/publishing")
async def get_publishing_intelligence(
    platform: Optional[str] = Query(None, description="Filter by platform"),
    min_observations: int = Query(
        20,
        ge=5,
        description="Minimum observations required to report a publishing window",
    ),
):
    """
    Observe publishing-time patterns from verified historical data.

    Analysis is based ONLY on actual recorded observation timestamps.
    If insufficient data, returns 'insufficient_evidence' status — NO fabrication.

    Returns per-platform:
    - best_hours: hours-of-day with above-average engagement (OBSERVED, not predicted)
    - best_days: days-of-week with above-average engagement (OBSERVED)
    - total_observations used
    - 'sufficient_evidence' boolean
    - note: explicit disclaimer about observed vs predicted
    """
    try:
        platforms = [platform] if platform and platform in VALID_PLATFORMS else VALID_PLATFORMS
        pool = _get_pool()
        results = []

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                for p in platforms:
                    cfg = _PLATFORM_CFG[p]
                    metric = cfg["primary_metric"]
                    id_col = cfg["id_col"]
                    table = cfg["history_table"]

                    # Deduplicate per content+hour to avoid duplication from re-ingestion
                    await cur.execute(
                        cast(
                            LiteralString,
                            f"""
                        WITH deduped AS (
                            SELECT
                                {id_col}::text AS content_id,
                                DATE_TRUNC('hour', collected_at) AS obs_hour,
                                MAX({metric}::bigint) AS metric,
                                MAX(likes::bigint) AS likes
                            FROM {table}
                            WHERE collected_at >= NOW() - INTERVAL '90 days'
                            GROUP BY {id_col}, obs_hour
                        ),
                        by_hour AS (
                            SELECT
                                EXTRACT(HOUR FROM obs_hour)::int AS hour_of_day,
                                AVG(metric) AS avg_metric,
                                AVG(likes) AS avg_likes,
                                COUNT(*) AS n_obs
                            FROM deduped
                            GROUP BY hour_of_day
                        ),
                        overall_avg AS (
                            SELECT AVG(avg_metric) AS grand_avg FROM by_hour
                        )
                        SELECT
                            b.hour_of_day,
                            ROUND(b.avg_metric::numeric, 1) AS avg_metric,
                            ROUND(b.avg_likes::numeric, 1) AS avg_likes,
                            b.n_obs,
                            ROUND((b.avg_metric / NULLIF(o.grand_avg, 0) - 1) * 100, 1) AS pct_above_avg
                        FROM by_hour b, overall_avg o
                        ORDER BY b.avg_metric DESC
                        """,
                        )
                    )
                    hour_rows = await cur.fetchall()
                    hour_cols = [d[0] for d in cur.description] if cur.description else []

                    await cur.execute(
                        cast(
                            LiteralString,
                            f"""
                        WITH deduped AS (
                            SELECT
                                {id_col}::text AS content_id,
                                DATE_TRUNC('hour', collected_at) AS obs_hour,
                                MAX({metric}::bigint) AS metric,
                                MAX(likes::bigint) AS likes
                            FROM {table}
                            WHERE collected_at >= NOW() - INTERVAL '90 days'
                            GROUP BY {id_col}, obs_hour
                        ),
                        by_day AS (
                            SELECT
                                EXTRACT(DOW FROM obs_hour)::int AS day_of_week,
                                AVG(metric) AS avg_metric,
                                AVG(likes) AS avg_likes,
                                COUNT(*) AS n_obs
                            FROM deduped
                            GROUP BY day_of_week
                        ),
                        overall_avg AS (
                            SELECT AVG(avg_metric) AS grand_avg FROM by_day
                        )
                        SELECT
                            b.day_of_week,
                            ROUND(b.avg_metric::numeric, 1) AS avg_metric,
                            ROUND(b.avg_likes::numeric, 1) AS avg_likes,
                            b.n_obs,
                            ROUND((b.avg_metric / NULLIF(o.grand_avg, 0) - 1) * 100, 1) AS pct_above_avg
                        FROM by_day b, overall_avg o
                        ORDER BY b.avg_metric DESC
                        """,
                        )
                    )
                    day_rows = await cur.fetchall()
                    day_cols = [d[0] for d in cur.description] if cur.description else []

                    # Count total observations for this platform
                    await cur.execute(
                        cast(
                            LiteralString,
                            f"SELECT COUNT(*) FROM {table} WHERE collected_at >= NOW() - INTERVAL '90 days'",
                        )
                    )
                    obs_row = await cur.fetchone()
                    total_obs = (obs_row[0] if obs_row else 0) or 0

                    sufficient = int(total_obs) >= min_observations

                    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

                    best_hours = []
                    for row in hour_rows[:5]:  # top 5 hours
                        d = dict(zip(hour_cols, row))
                        best_hours.append({
                            "hour": int(d["hour_of_day"]),
                            "label": f"{int(d['hour_of_day']):02d}:00",
                            "avg_metric": float(d["avg_metric"] or 0),
                            "avg_likes": float(d["avg_likes"] or 0),
                            "n_obs": int(d["n_obs"] or 0),
                            "pct_above_avg": float(d["pct_above_avg"] or 0),
                        })

                    best_days = []
                    for row in day_rows[:4]:  # top 4 days
                        d = dict(zip(day_cols, row))
                        dow = int(d["day_of_week"])
                        best_days.append({
                            "day_of_week": dow,
                            "label": day_names[dow],
                            "avg_metric": float(d["avg_metric"] or 0),
                            "avg_likes": float(d["avg_likes"] or 0),
                            "n_obs": int(d["n_obs"] or 0),
                            "pct_above_avg": float(d["pct_above_avg"] or 0),
                        })

                    results.append({
                        "platform": p,
                        "sufficient_evidence": sufficient,
                        "total_observations": int(total_obs),
                        "min_observations_required": min_observations,
                        "note": (
                            "OBSERVED: These windows reflect when your content historically received "
                            "above-average engagement based on actual collected timestamps. "
                            "This is NOT a prediction — it is a pattern observed from verified records."
                        ) if sufficient else (
                            f"INSUFFICIENT EVIDENCE: Only {int(total_obs)} observations collected "
                            f"(minimum {min_observations} required). "
                            "No publishing window recommendation can be made from this data."
                        ),
                        "best_hours": best_hours if sufficient else [],
                        "best_days": best_days if sufficient else [],
                    })

        from app.ml.readiness import get_xgboost_status_detail
        xgb_platform = platforms[0] if len(platforms) == 1 else None
        xgboost_status = get_xgboost_status_detail(xgb_platform)

        return {
            "ok": True,
            "publishing_intelligence": results,
            "xgboost_status": xgboost_status,
        }

    except Exception as exc:
        logger.error("[/intelligence/publishing] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch publishing intelligence.")# ---------------------------------------------------------------------------
# POST /intelligence/chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    platform: Optional[str] = None


_GROUNDING_QUERIES = {
    "surge": """
        SELECT platform, content_id, title, classification, velocity_ratio, evidence_reason, analyzed_at
        FROM ai_suggestions
        WHERE classification IN ('BOOMING_SURGE', 'SURGE_CANDIDATE')
        ORDER BY analyzed_at DESC, velocity_ratio DESC NULLS LAST
        LIMIT 5
    """,
    "low_performing": """
        SELECT platform, content_id, title, classification, velocity_ratio, baseline_metric, evidence_reason, analyzed_at
        FROM ai_suggestions
        WHERE classification = 'LOW_PERFORMING'
        ORDER BY analyzed_at DESC
        LIMIT 5
    """,
    "total_count": """
        WITH latest AS (
            SELECT DISTINCT ON (platform, content_id) classification
            FROM ai_suggestions
            WHERE classification = ANY(ARRAY['BOOMING_SURGE','SURGE_CANDIDATE','ELEVATED','LOW_PERFORMING'])
            ORDER BY platform, content_id, analyzed_at DESC
        )
        SELECT classification, COUNT(*) AS cnt
        FROM latest
        GROUP BY classification
        ORDER BY cnt DESC
    """,
    "top_content": """
        SELECT DISTINCT ON (platform, content_id)
            platform, content_id, title, classification, current_metric, velocity_ratio, analyzed_at
        FROM ai_suggestions
        WHERE classification IN ('BOOMING_SURGE','SURGE_CANDIDATE')
        ORDER BY platform, content_id, analyzed_at DESC, velocity_ratio DESC NULLS LAST
        LIMIT 5
    """,
}


def _classify_question(question: str) -> str:
    """Map user question to a grounding query key."""
    q = question.lower()
    if any(w in q for w in ["surge", "surging", "trending", "viral", "boom"]):
        return "surge"
    if any(w in q for w in ["low", "performing", "underperform", "poor", "struggle"]):
        return "low_performing"
    if any(w in q for w in ["how many", "total", "count", "number of"]):
        return "total_count"
    if any(w in q for w in ["top", "best", "highest", "most view", "most reach"]):
        return "top_content"
    return "total_count"


def _format_grounded_answer(question: str, query_key: str, rows: list[dict]) -> str:
    """Compose a factual answer strictly from verified records."""
    if not rows:
        return (
            "No verified records match your question in the current AI Insights dataset. "
            "Run a fresh AI scan to populate the latest data."
        )

    q = question.lower()

    if query_key == "total_count":
        lines = ["**Verified AI Insights Counts** (from database as of last scan):"]
        for row in rows:
            cls = row.get("classification", "")
            cnt = row.get("cnt", 0)
            lines.append(f"- **{cls}**: {cnt} content item(s)")
        lines.append("\n*Source: ai_suggestions table, latest record per content item.*")
        return "\n".join(lines)

    if query_key == "surge":
        lines = ["**Currently Surging Content** (verified from last scan):"]
        for row in rows:
            title = row.get("title") or row.get("content_id", "")
            platform = row.get("platform", "")
            cls = row.get("classification", "")
            vr = row.get("velocity_ratio")
            reason = row.get("evidence_reason", "")
            vr_str = f" — velocity ratio {float(vr):.2f}x" if vr is not None else ""
            lines.append(f"- **{title}** [{platform}] ({cls}{vr_str})")
            if reason:
                lines.append(f"  {reason}")
        lines.append("\n*Source: ai_suggestions table. Velocity ratios are computed from real observation history.*")
        return "\n".join(lines)

    if query_key == "low_performing":
        lines = ["**Low Performing Content** (verified from last scan):"]
        for row in rows:
            title = row.get("title") or row.get("content_id", "")
            platform = row.get("platform", "")
            reason = row.get("evidence_reason", "")
            lines.append(f"- **{title}** [{platform}]")
            if reason:
                lines.append(f"  {reason}")
        lines.append("\n*Source: ai_suggestions table. LOW_PERFORMING requires both sub-baseline velocity and sufficient coverage history.*")
        return "\n".join(lines)

    if query_key == "top_content":
        lines = ["**Top Performing Content** (by velocity ratio, verified from last scan):"]
        for row in rows:
            title = row.get("title") or row.get("content_id", "")
            platform = row.get("platform", "")
            cls = row.get("classification", "")
            vr = row.get("velocity_ratio")
            metric = row.get("current_metric")
            vr_str = f", velocity {float(vr):.2f}x" if vr is not None else ""
            metric_str = f", {int(metric):,} views/reach" if metric is not None else ""
            lines.append(f"- **{title}** [{platform}] ({cls}{vr_str}{metric_str})")
        lines.append("\n*Source: ai_suggestions table.*")
        return "\n".join(lines)

    return "I can only answer questions grounded in verified AI Insights database records."


@router.post("/chat")
async def intelligence_chat(body: ChatRequest):
    """
    Grounded chatbot: answers factual questions strictly from verified DB records.

    NEVER fabricates data. If the data does not support an answer, explicitly states so.
    No LLM is used here — answers are constructed from live DB queries.
    """
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 chars).")

    # Sanitize: strip any injection patterns
    question_safe = re.sub(r"[;\'\"\-\-]", " ", question)

    try:
        pool = _get_pool()
        query_key = _classify_question(question_safe)
        sql = _GROUNDING_QUERIES[query_key]

        # Apply platform filter if specified and valid
        platform_filter = ""
        params: list = []
        if body.platform and body.platform in VALID_PLATFORMS:
            if "FROM ai_suggestions" in sql:
                # Inject platform filter into WHERE clause
                if "WHERE" in sql:
                    sql = sql.replace("ORDER BY", f"AND platform = %s\n        ORDER BY", 1)
                else:
                    sql = sql.replace("ORDER BY", f"WHERE platform = %s\n        ORDER BY", 1)
                params = [body.platform]

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(cast(LiteralString, sql), params)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = [dict(zip(cols, row)) for row in await cur.fetchall()]

        # Serialize datetime fields
        for row in rows:
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
                elif hasattr(v, "__float__"):
                    row[k] = float(v)

        answer = _format_grounded_answer(question_safe, query_key, rows)

        return {
            "ok": True,
            "question": question,
            "answer": answer,
            "grounding_query": query_key,
            "records_used": len(rows),
            "disclaimer": (
                "This answer is grounded exclusively in verified records from the AI Insights database. "
                "No fabrication or interpolation. Data reflects the last completed AI scan."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[/intelligence/chat] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process chat query.")

"""
Intelligence API routes.

GET  /intelligence/performance   — Performance chart data (deduplicated daily observations)
GET  /intelligence/publishing    — Observed publishing time analysis per platform
POST /intelligence/chat          — AI Assistant: grounded evidence retrieval → LLM → validated response

All data is derived strictly from verified records in ai_suggestions and *_history_ai tables.
No fabrication, interpolation, or prediction presented as observed fact.
The chat endpoint retrieves verified evidence, passes it to the configured LLM provider, and
returns the grounded LLM response. The LLM never reads the database directly.
"""
from __future__ import annotations

import logging
import re
from datetime import timezone
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
        raise HTTPException(status_code=500, detail="Failed to fetch publishing intelligence.")


# ---------------------------------------------------------------------------
# POST /intelligence/chat  — Real LLM AI Assistant
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    platform: Optional[str] = None
    # Bounded conversation history for follow-up context.
    # Client sends the last N turns; server never stores session state.
    # Max 6 turns (3 user + 3 assistant) to keep context bounded.
    conversation_history: Optional[list[dict]] = None


# ── Intent routing ─────────────────────────────────────────────────────────
# Maps the question to the right evidence retrieval query.
# Intent classification determines WHICH evidence to retrieve — NOT a canned answer.

def _classify_intent(question: str) -> str:
    """
    Classify the question intent to select the right evidence query.
    Returns one of: surge | low_performing | publishing | specific_content |
                    platform_overview | xgboost | total_count
    """
    q = question.lower()
    # Specific content query — look for quoted strings or 'this video/content'
    if any(w in q for w in ["this video", "this content", "this post", "this reel", "why is"]):
        return "specific_content"
    if any(w in q for w in ["surge", "surging", "trending", "viral", "boom", "spike"]):
        return "surge"
    if any(w in q for w in ["low", "performing", "underperform", "poor", "struggle", "stalled", "dropping"]):
        return "low_performing"
    if any(w in q for w in ["publish", "when should", "best time", "post time", "schedule", "timing"]):
        return "publishing"
    if any(w in q for w in ["xgboost", "ml", "machine learning", "predict", "model"]):
        return "xgboost"
    if any(w in q for w in ["what is happening", "overview", "summary", "platform", "youtube", "instagram", "facebook"]):
        return "platform_overview"
    if any(w in q for w in ["how many", "total", "count", "number of", "top", "best", "highest", "most view", "next", "do next", "recommend"]):
        return "top_content"
    return "platform_overview"


async def _retrieve_evidence(
    intent: str,
    platform: Optional[str],
    question: str,
) -> dict:
    """
    Retrieve verified evidence from the database for the classified intent.

    Returns a dict with:
        - intent: the classified intent
        - records: list of verified DB rows
        - extra: optional additional context (e.g. publishing windows, xgboost status)
        - record_count: number of records retrieved
    """
    pool = _get_pool()
    valid_platforms = list(_PLATFORM_CFG.keys())
    plat_filter = platform if (platform and platform in valid_platforms) else None

    records: list[dict] = []
    extra: dict = {}

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                if intent == "surge":
                    sql = """
                        SELECT platform, content_id, title, classification,
                               velocity_ratio, like_acceleration, evidence_reason,
                               ai_recommendation, coverage_hours, analyzed_at
                        FROM ai_suggestions
                        WHERE classification IN ('BOOMING_SURGE', 'SURGE_CANDIDATE')
                        {}
                        ORDER BY analyzed_at DESC, velocity_ratio DESC NULLS LAST
                        LIMIT 8
                    """.format("AND platform = %s" if plat_filter else "")
                    await cur.execute(
                        cast(LiteralString, sql),
                        [plat_filter] if plat_filter else [],
                    )

                elif intent == "low_performing":
                    sql = """
                        SELECT platform, content_id, title, classification,
                               velocity_ratio, baseline_metric, evidence_reason,
                               ai_recommendation, coverage_hours, analyzed_at
                        FROM ai_suggestions
                        WHERE classification = 'LOW_PERFORMING'
                        {}
                        ORDER BY analyzed_at DESC
                        LIMIT 8
                    """.format("AND platform = %s" if plat_filter else "")
                    await cur.execute(
                        cast(LiteralString, sql),
                        [plat_filter] if plat_filter else [],
                    )

                elif intent == "publishing":
                    # Retrieve best observed publishing windows per platform
                    sql = """
                        SELECT platform, content_id, title, classification,
                               velocity_ratio, evidence_reason, analyzed_at
                        FROM ai_suggestions
                        WHERE classification IN ('BOOMING_SURGE', 'SURGE_CANDIDATE', 'ELEVATED')
                        {}
                        ORDER BY analyzed_at DESC, velocity_ratio DESC NULLS LAST
                        LIMIT 5
                    """.format("AND platform = %s" if plat_filter else "")
                    await cur.execute(
                        cast(LiteralString, sql),
                        [plat_filter] if plat_filter else [],
                    )
                    # Also fetch publishing hour intelligence for context
                    platforms_to_check = [plat_filter] if plat_filter else valid_platforms
                    pub_windows: list[dict] = []
                    for p in platforms_to_check[:2]:  # limit to 2 platforms for context size
                        cfg = _PLATFORM_CFG[p]
                        metric = cfg["primary_metric"]
                        id_col = cfg["id_col"]
                        table = cfg["history_table"]
                        await cur.execute(
                            cast(
                                LiteralString,
                                f"""
                                WITH deduped AS (
                                    SELECT {id_col}::text AS content_id,
                                           DATE_TRUNC('hour', collected_at) AS obs_hour,
                                           MAX({metric}::bigint) AS metric
                                    FROM {table}
                                    WHERE collected_at >= NOW() - INTERVAL '60 days'
                                    GROUP BY {id_col}, obs_hour
                                ),
                                by_hour AS (
                                    SELECT EXTRACT(HOUR FROM obs_hour)::int AS h,
                                           AVG(metric) AS avg_m, COUNT(*) AS n
                                    FROM deduped GROUP BY h
                                ),
                                avg_all AS (SELECT AVG(avg_m) AS grand_avg FROM by_hour)
                                SELECT b.h, ROUND(b.avg_m::numeric,1) AS avg_m, b.n,
                                       ROUND((b.avg_m/NULLIF(a.grand_avg,0)-1)*100,1) AS pct
                                FROM by_hour b, avg_all a
                                ORDER BY b.avg_m DESC LIMIT 3
                                """,
                            )
                        )
                        hour_rows = await cur.fetchall()
                        if hour_rows:
                            pub_windows.append({
                                "platform": p,
                                "top_hours": [
                                    {"hour": r[0], "pct_above_avg": float(r[3] or 0), "n_obs": r[2]}
                                    for r in hour_rows
                                ],
                            })
                    extra["publishing_windows"] = pub_windows

                elif intent == "xgboost":
                    from app.ml.xgboost_service import get_xgboost_api_status
                    extra["xgboost_status"] = get_xgboost_api_status()
                    sql = """
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id) classification
                            FROM ai_suggestions
                            WHERE classification = ANY(ARRAY['BOOMING_SURGE','SURGE_CANDIDATE','ELEVATED','LOW_PERFORMING'])
                            ORDER BY platform, content_id, analyzed_at DESC
                        )
                        SELECT classification, COUNT(*) AS cnt
                        FROM latest GROUP BY classification ORDER BY cnt DESC
                    """
                    await cur.execute(cast(LiteralString, sql))

                elif intent == "specific_content":
                    sql = """
                        SELECT platform, content_id, title, classification,
                               velocity_ratio, like_acceleration, current_metric,
                               baseline_metric, evidence_reason,
                               ai_recommendation, structured_analysis,
                               coverage_hours, llm_status, analyzed_at
                        FROM ai_suggestions
                        {}
                        ORDER BY analyzed_at DESC
                        LIMIT 5
                    """.format("WHERE platform = %s" if plat_filter else "WHERE 1=1")
                    await cur.execute(
                        cast(LiteralString, sql),
                        [plat_filter] if plat_filter else [],
                    )

                else:
                    # platform_overview / total_count / top_content / next action
                    sql = """
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id)
                                platform, content_id, title, classification,
                                velocity_ratio, current_metric, evidence_reason,
                                ai_recommendation, analyzed_at
                            FROM ai_suggestions
                            WHERE classification IN ('BOOMING_SURGE','SURGE_CANDIDATE','ELEVATED','LOW_PERFORMING')
                            {}
                            ORDER BY platform, content_id, analyzed_at DESC
                        )
                        SELECT * FROM latest
                        ORDER BY
                            CASE classification
                                WHEN 'BOOMING_SURGE' THEN 1
                                WHEN 'SURGE_CANDIDATE' THEN 2
                                WHEN 'LOW_PERFORMING' THEN 3
                                ELSE 4
                            END,
                            velocity_ratio DESC NULLS LAST
                        LIMIT 10
                    """.format("AND platform = %s" if plat_filter else "")
                    await cur.execute(
                        cast(LiteralString, sql),
                        [plat_filter] if plat_filter else [],
                    )

                if cur.description:
                    cols = [d[0] for d in cur.description]
                    for row in await cur.fetchall():
                        d = dict(zip(cols, row))
                        for k, v in d.items():
                            if hasattr(v, "isoformat"):
                                d[k] = v.isoformat()
                            elif hasattr(v, "__float__") and not isinstance(v, (int, bool)):
                                d[k] = float(v)
                        records.append(d)

    except Exception as exc:
        logger.error("[/intelligence/chat] Evidence retrieval error: %s", exc)
        raise

    return {
        "intent": intent,
        "records": records,
        "extra": extra,
        "record_count": len(records),
    }


def _build_chat_system_prompt() -> str:
    return (
        "You are an AI assistant for a Tamil media analytics dashboard. "
        "You answer questions about content performance using ONLY the verified data provided below. "
        "You MUST NOT invent, fabricate, or extrapolate any metrics, percentages, content titles, "
        "dates, classifications, causes, or recommendations beyond what is in the supplied evidence. "
        "If the evidence is insufficient to answer the question, say so explicitly. "
        "Distinguish clearly between OBSERVED FACTS (from the database) and INTERPRETATION. "
        "Never claim causation — use language like 'may indicate', 'consistent with', 'observed pattern'. "
        "Keep responses concise and actionable. Format lists with bullet points where appropriate. "
        "Do not mention SQL, databases, or internal system details in your response. "
        "Your audience is the CEO of a media company."
    )


def _build_chat_user_prompt(
    question: str,
    evidence: dict,
    conversation_history: Optional[list[dict]],
) -> str:
    intent = evidence["intent"]
    records = evidence["records"]
    extra = evidence.get("extra", {})

    lines = ["=== VERIFIED EVIDENCE FROM DATABASE ==="]

    if intent == "xgboost":
        xgb = extra.get("xgboost_status", {})
        lines.append(f"XGBoost Pipeline Status:")
        lines.append(f"  data_prerequisites_met: {xgb.get('data_prerequisites_met', False)}")
        lines.append(f"  model_trained: {xgb.get('model_trained', False)}")
        lines.append(f"  model_loaded: {xgb.get('model_loaded', False)}")
        lines.append(f"  prediction_available: {xgb.get('prediction_available', False)}")
        lines.append(f"  qualification_status: {xgb.get('qualification_status', 'not_loaded')}")
        lines.append(f"  qualification_reason: {xgb.get('qualification_reason', '')}")
        lines.append(f"  Note: {xgb.get('note', '')}")

    if not records and intent != "xgboost":
        lines.append("No matching records found in the current AI Insights dataset.")
        lines.append("Tell the user to run a fresh AI scan to populate latest data.")
    else:
        lines.append(f"Records retrieved: {len(records)} ({intent} intent)")
        for i, r in enumerate(records[:8], 1):
            lines.append(f"\n[{i}] {r.get('title') or r.get('content_id', 'Unknown')}")
            lines.append(f"    Platform: {r.get('platform', '?')}")
            lines.append(f"    Classification: {r.get('classification', '?')}")
            if r.get("velocity_ratio") is not None:
                lines.append(f"    Velocity ratio: {r['velocity_ratio']:.3f}x")
            if r.get("like_acceleration") is not None:
                lines.append(f"    Like acceleration: {r['like_acceleration']:.2f}%")
            if r.get("current_metric") is not None:
                lines.append(f"    Current metric: {r['current_metric']:,}")
            if r.get("baseline_metric") is not None:
                lines.append(f"    Hourly baseline: {r['baseline_metric']:.3f}")
            if r.get("coverage_hours") is not None:
                lines.append(f"    History coverage: {r['coverage_hours']:.1f}h")
            if r.get("evidence_reason"):
                lines.append(f"    Evidence: {r['evidence_reason']}")
            if r.get("ai_recommendation") and r["ai_recommendation"] not in (
                "Recommendation unavailable",
                "Recommendation unavailable (validation failed)",
            ):
                lines.append(f"    AI Recommendation: {r['ai_recommendation'][:200]}")
            if r.get("analyzed_at"):
                lines.append(f"    Last analyzed: {r['analyzed_at']}")

    if extra.get("publishing_windows"):
        lines.append("\n=== OBSERVED PUBLISHING WINDOWS (from history) ===")
        for pw in extra["publishing_windows"]:
            lines.append(f"Platform: {pw['platform']}")
            for h in pw.get("top_hours", []):
                lines.append(
                    f"  Hour {h['hour']:02d}:00 — {h['pct_above_avg']:+.0f}% above avg ({h['n_obs']} observations)"
                )
        lines.append("NOTE: These are OBSERVED historical patterns, NOT predictions of future performance.")

    lines.append("\n=== QUESTION ===")
    lines.append(question)

    if conversation_history:
        # Append last 4 turns of context (bounded)
        bounded = conversation_history[-4:]
        lines.append("\n=== RECENT CONVERSATION CONTEXT ===")
        for turn in bounded:
            role = turn.get("role", "")
            text = (turn.get("content") or turn.get("text") or "")[:300]
            if role and text:
                lines.append(f"{role.upper()}: {text}")

    lines.append("\nAnswer the question using ONLY the verified evidence above.")
    return "\n".join(lines)


@router.post("/chat")
async def intelligence_chat(body: ChatRequest):
    """
    AI Assistant: grounded evidence retrieval → LLM → validated response.

    Flow:
      1. Classify question intent (surge / low_performing / publishing / etc.)
      2. Retrieve relevant verified evidence from ai_suggestions + *_history_ai
      3. Build a bounded evidence package (no raw tables, no full history dumps)
      4. Call the configured LLM provider (LM Studio in dev, Bedrock in prod)
      5. Return the grounded LLM response

    The LLM never accesses PostgreSQL directly — it only receives the evidence
    package assembled from verified DB records.
    If the provider is unavailable, returns a provider_unavailable status rather
    than fabricating an answer.
    """
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 chars).")

    # Sanitize question — strip SQL injection patterns before classification
    question_safe = re.sub(r"[;\'\"\-\-]", " ", question)

    # Validate and bound conversation history
    history: Optional[list[dict]] = None
    if body.conversation_history:
        # Accept at most 6 turns, each bounded to 500 chars
        history = [
            {
                "role": t.get("role", "")[:20],
                "content": (t.get("content") or t.get("text") or "")[:500],
            }
            for t in body.conversation_history[-6:]
            if t.get("role") in ("user", "assistant")
        ]

    try:
        # ── 1. Classify intent ───────────────────────────────────────────
        intent = _classify_intent(question_safe)
        logger.info("[chat] intent=%s platform=%s question_len=%d", intent, body.platform, len(question))

        # ── 2. Retrieve verified evidence ────────────────────────────────
        evidence = await _retrieve_evidence(intent, body.platform, question_safe)

        # ── 3. Build prompts ─────────────────────────────────────────────
        system_prompt = _build_chat_system_prompt()
        user_prompt = _build_chat_user_prompt(question_safe, evidence, history)

        # ── 4. Call LLM provider ─────────────────────────────────────────
        from app.providers import get_provider_with_fallback
        from app.core.exceptions import ProviderUnavailableError, ProviderError

        provider_name = "unknown"
        llm_answer: Optional[str] = None
        provider_status = "ok"

        try:
            primary_provider, fallback_provider = get_provider_with_fallback()
            provider_meta = primary_provider.get_provider_metadata()
            provider_name = provider_meta.get("provider", "unknown")

            # Build a minimal EvidencePackage-like call using the raw chat interface.
            # We send the prompts directly via the provider's underlying HTTP/boto client
            # rather than through generate_structured_analysis (which expects EditorialAnalysis JSON).
            try:
                llm_answer = await _call_provider_chat(
                    provider=primary_provider,
                    provider_name=provider_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except ProviderUnavailableError as primary_exc:
                logger.warning("[chat] Primary provider unavailable: %s", primary_exc)
                if fallback_provider is not None:
                    fb_meta = fallback_provider.get_provider_metadata()
                    fb_name = fb_meta.get("provider", "fallback")
                    logger.info("[chat] Trying fallback provider: %s", fb_name)
                    llm_answer = await _call_provider_chat(
                        provider=fallback_provider,
                        provider_name=fb_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                    )
                    provider_name = fb_name
                else:
                    raise  # no fallback — propagate to outer except
        except ProviderUnavailableError as exc:
            logger.warning("[chat] Provider unavailable: %s", exc)
            provider_status = "unavailable"
            llm_answer = None
        except ProviderError as exc:
            logger.warning("[chat] Provider error: %s", exc)
            provider_status = "error"
            llm_answer = None
        except Exception as exc:
            logger.error("[chat] Unexpected provider error: %s", exc)
            provider_status = "error"
            llm_answer = None

        # ── 5. Build response ────────────────────────────────────────────
        if provider_status != "ok" or not llm_answer:
            # Provider unavailable — return structured evidence without LLM
            # so the frontend can show a clear state rather than silence
            fallback = _build_fallback_answer(evidence)
            return {
                "ok": True,
                "question": question,
                "answer": fallback,
                "intent": intent,
                "records_used": evidence["record_count"],
                "provider_status": provider_status,
                "provider": provider_name,
                "llm_used": False,
                "disclaimer": (
                    "LLM provider currently unavailable. "
                    "Answer below is derived directly from verified database records without LLM interpretation. "
                    "Data reflects the last completed AI scan."
                ),
            }

        return {
            "ok": True,
            "question": question,
            "answer": llm_answer,
            "intent": intent,
            "records_used": evidence["record_count"],
            "provider_status": "ok",
            "provider": provider_name,
            "llm_used": True,
            "disclaimer": (
                "Answer grounded in verified records from the AI Insights database. "
                "The LLM received only pre-retrieved, verified evidence — it did not access "
                "the database directly. Data reflects the last completed AI scan."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[/intelligence/chat] Error: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to process chat query.")


async def _call_provider_chat(
    provider,
    provider_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Call the provider's underlying HTTP/boto client for a free-form chat response.

    We bypass generate_structured_analysis (which requires EditorialAnalysis JSON schema)
    and call the raw completion endpoint directly, since chat answers are plain text.
    """
    from app.core.config import get_settings
    from app.core.exceptions import ProviderUnavailableError, ProviderError

    settings = get_settings()

    if provider_name == "lmstudio":
        import httpx
        payload = {
            "model": settings.LMSTUDIO_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.LMSTUDIO_TEMPERATURE,
            "max_tokens": settings.LMSTUDIO_MAX_TOKENS,
        }
        endpoint = f"{settings.LMSTUDIO_BASE_URL}/chat/completions"
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(settings.LMSTUDIO_TIMEOUT_SECONDS),
            write=10.0,
            pool=10.0,
        )
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Cannot connect to LM Studio at {settings.LMSTUDIO_BASE_URL}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"LM Studio timed out after {settings.LMSTUDIO_TIMEOUT_SECONDS}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"LM Studio HTTP {exc.response.status_code}"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(f"LM Studio request error: {exc}") from exc

        body = response.json()
        return body["choices"][0]["message"]["content"].strip()

    elif provider_name == "bedrock":
        import asyncio
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

        session_kwargs: dict = {}
        if settings.AWS_ACCESS_KEY_ID:
            session_kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        if settings.AWS_SECRET_ACCESS_KEY:
            session_kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN

        boto_config = Config(
            connect_timeout=10,
            read_timeout=float(settings.BEDROCK_TIMEOUT_SECONDS),
            retries={"max_attempts": 2},
        )
        session = boto3.Session(**session_kwargs)
        client = session.client(
            "bedrock-runtime",
            region_name=settings.AWS_REGION,
            config=boto_config,
        )

        # Use the same model-ID remapping as BedrockProvider to ensure the
        # global. cross-region inference prefix is applied when required.
        # The bare "amazon.nova-2-lite-v1:0" fails with ValidationException
        # in ap-south-1; the "global." prefix is required for cross-region calls.
        model_id = settings.BEDROCK_MODEL_ID
        if model_id and not model_id.startswith("global.") and model_id.startswith("amazon.nova-"):
            model_id = f"global.{model_id}"

        payload = {
            "modelId": model_id,
            "system": [{"text": system_prompt}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {
                "maxTokens": settings.BEDROCK_MAX_TOKENS,
                "temperature": settings.BEDROCK_TEMPERATURE,
            },
        }
        try:
            response = await asyncio.to_thread(client.converse, **payload)
        except NoCredentialsError as exc:
            raise ProviderUnavailableError("AWS credentials not found.") from exc
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            msg = exc.response["Error"].get("Message", "")
            logger.error(
                "[chat] Bedrock ClientError: code=%s region=%s model=%s msg=%s",
                code, settings.AWS_REGION, model_id, msg[:200],
            )
            raise ProviderUnavailableError(
                f"Bedrock {code} (region={settings.AWS_REGION} model={model_id})"
            ) from exc
        except BotoCoreError as exc:
            raise ProviderUnavailableError(
                f"Bedrock connectivity error: {type(exc).__name__}"
            ) from exc

        try:
            return response["output"]["message"]["content"][0]["text"].strip()
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                f"Unexpected Bedrock response structure: {exc}"
            ) from exc

    else:
        raise ProviderUnavailableError(
            f"Unknown provider '{provider_name}' — cannot generate chat response."
        )


def _build_fallback_answer(evidence: dict) -> str:
    """
    Build a plain-text answer from raw evidence when the LLM is unavailable.
    This is a structured summary, NOT a canned template — it uses real retrieved data.
    """
    intent = evidence["intent"]
    records = evidence["records"]
    extra = evidence.get("extra", {})

    if not records and intent != "xgboost":
        return (
            "No verified records match this question in the current AI Insights dataset. "
            "Run a fresh AI scan to populate the latest data."
        )

    lines = ["**AI Insights** (AI reasoning temporarily unavailable — verified dashboard data below):"]

    if intent == "xgboost":
        xgb = extra.get("xgboost_status", {})
        lines.append(f"- Data prerequisites met: {xgb.get('data_prerequisites_met', False)}")
        lines.append(f"- Model trained: {xgb.get('model_trained', False)}")
        lines.append(f"- Model loaded: {xgb.get('model_loaded', False)}")
        lines.append(f"- Predictions active: {xgb.get('prediction_available', False)}")
        lines.append(f"- Status: {xgb.get('qualification_reason', 'Not configured')}")
        return "\n".join(lines)

    for r in records[:6]:
        title = r.get("title") or r.get("content_id", "Unknown")
        cls = r.get("classification", "")
        platform = r.get("platform", "")
        vr = r.get("velocity_ratio")
        vr_str = f" (velocity {float(vr):.2f}x)" if vr is not None else ""

        # Use classification-specific description, not the raw evidence_reason
        # which can contain stale or contradictory text from a prior scan.
        cls_desc: dict[str, str] = {
            "BOOMING_SURGE": "🚀 Confirmed surge — receiving strongly above-baseline engagement",
            "SURGE_CANDIDATE": "📈 Emerging surge signal — above-baseline velocity, still accumulating history",
            "ELEVATED": "↑ Above-average traction — moderately outperforming baseline",
            "LOW_PERFORMING": "⚠️ Sub-baseline performance — engagement stalled below historical average",
            "NOMINAL": "→ Within normal baseline range",
        }
        desc = cls_desc.get(cls, cls)
        lines.append(f"- **{title}** [{platform}]{vr_str}")
        lines.append(f"  {desc}")

    if extra.get("publishing_windows"):
        lines.append("\n**Observed publishing windows:**")
        for pw in extra["publishing_windows"]:
            hours_str = ", ".join(
                f"{h['hour']:02d}:00 (+{h['pct_above_avg']:.0f}%)"
                for h in pw.get("top_hours", [])
            )
            lines.append(f"- {pw['platform']}: {hours_str}")
        lines.append("*(Based on actual observed timestamps — not predictions)*")

    lines.append("\n*Source: ai_suggestions table. Data reflects the last completed AI scan.*")
    return "\n".join(lines)

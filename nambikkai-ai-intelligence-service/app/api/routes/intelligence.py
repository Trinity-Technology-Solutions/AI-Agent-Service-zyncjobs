"""
Intelligence API routes.

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

        return {
            "ok": True,
            "publishing_intelligence": results,
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
    period: Optional[str] = None
    content_type: Optional[str] = None
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
    Returns one of: casual | specific_content | high_performing | low_performing | publishing | platform_overview
    """
    q = question.lower().strip()

    # Casual conversation — do not query analytics for these
    casual_patterns = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "good night", "thanks", "thank you", "thank", "bye", "goodbye", "see you",
        "cheers", "great", "ok", "okay", "sure", "got it", "sounds good",
        "how are you", "what can you do", "who are you", "what are you",
        "help me", "what do you do",
    ]
    for pat in casual_patterns:
        if q == pat or q.startswith(pat + " ") or q.endswith(" " + pat) or q == pat + "!":
            return "casual"
    # Short greetings / thanks with punctuation
    if len(q) <= 15 and any(w in q for w in ["hi", "hello", "hey", "thanks", "thank", "bye"]):
        return "casual"

    if any(w in q for w in ["this video", "this content", "this post", "this reel", "why is"]):
        return "specific_content"
    if any(w in q for w in ["high", "surge", "surging", "trending", "viral", "boom", "spike", "top", "best", "strongest", "leading"]):
        return "high_performing"
    if any(w in q for w in ["low", "underperform", "poor", "struggle", "stalled", "dropping", "attention", "improve", "fix", "needs", "worst", "weakest"]):
        return "low_performing"
    if any(w in q for w in ["publish", "when should", "best time", "post time", "schedule", "timing"]):
        return "publishing"
    return "platform_overview"


async def _retrieve_evidence(
    intent: str,
    platform: Optional[str],
    question: str,
    period: Optional[str] = None,
    content_type: Optional[str] = None,
) -> dict:
    """
    Retrieve verified evidence from the database for the classified intent.

    Returns a dict with:
        - intent: the classified intent
        - records: list of verified DB rows
        - extra: optional additional context
        - record_count: number of records retrieved
    """
    pool = _get_pool()
    valid_platforms = list(_PLATFORM_CFG.keys())
    plat_filter = platform if (platform and platform in valid_platforms) else None

    q_lower = question.lower()
    # Content type inference
    inferred_content_type: Optional[str] = None
    if content_type and content_type.lower() != "all":
        inferred_content_type = f"%{content_type}%"
    else:
        if "short" in q_lower:
            inferred_content_type = "%Short%"
        elif "reel" in q_lower:
            inferred_content_type = "%Reel%"
        elif "post" in q_lower:
            inferred_content_type = "%Post%"

    # Period inference
    inferred_period: Optional[str] = None
    if period and period.lower() != "all":
        inferred_period = period.lower()
    else:
        if "7d" in q_lower or "7 day" in q_lower or "seven day" in q_lower:
            inferred_period = "7d"
        elif "90d" in q_lower or "90 day" in q_lower:
            inferred_period = "90d"
        elif "30d" in q_lower or "30 day" in q_lower or "month" in q_lower:
            inferred_period = "30d"

    # Platform inference if not provided
    if not plat_filter:
        if "youtube" in q_lower:
            plat_filter = "youtube"
        elif "instagram" in q_lower or "insta" in q_lower:
            plat_filter = "instagram"
        elif "facebook" in q_lower or "fb" in q_lower:
            plat_filter = "facebook"

    records: list[dict] = []
    extra: dict = {
        "platform_filter": plat_filter,
        "content_type_filter": inferred_content_type,
        "period_filter": inferred_period,
    }

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                if intent == "high_performing":
                    clauses = ["classification = 'HIGH_PERFORMING'"]
                    params = []
                    if plat_filter:
                        clauses.append("platform = %s")
                        params.append(plat_filter)
                    if inferred_content_type:
                        clauses.append("content_type ILIKE %s")
                        params.append(inferred_content_type)
                    if inferred_period:
                        clauses.append("report_period = %s")
                        params.append(inferred_period)
                    sql = f"""
                        SELECT platform, content_id, title, content_type, canonical_url,
                               classification, current_metric, baseline_metric, metric_name,
                               likes, comments, peer_explanation, evidence_reason,
                               ai_recommendation, report_period, analyzed_at
                        FROM ai_suggestions
                        WHERE {" AND ".join(clauses)}
                        ORDER BY analyzed_at DESC, current_metric DESC NULLS LAST
                        LIMIT 8
                    """
                    await cur.execute(cast(LiteralString, sql), params)

                elif intent == "low_performing":
                    clauses = ["classification = 'LOW_PERFORMING'"]
                    params = []
                    if plat_filter:
                        clauses.append("platform = %s")
                        params.append(plat_filter)
                    if inferred_content_type:
                        clauses.append("content_type ILIKE %s")
                        params.append(inferred_content_type)
                    if inferred_period:
                        clauses.append("report_period = %s")
                        params.append(inferred_period)
                    sql = f"""
                        SELECT platform, content_id, title, content_type, canonical_url,
                               classification, current_metric, baseline_metric, metric_name,
                               likes, comments, peer_explanation, evidence_reason,
                               ai_recommendation, report_period, analyzed_at
                        FROM ai_suggestions
                        WHERE {" AND ".join(clauses)}
                        ORDER BY analyzed_at DESC, current_metric ASC NULLS LAST
                        LIMIT 8
                    """
                    await cur.execute(cast(LiteralString, sql), params)

                elif intent == "publishing":
                    sql = """
                        SELECT platform, content_id, title, content_type, canonical_url,
                               classification, current_metric, baseline_metric,
                               evidence_reason, analyzed_at
                        FROM ai_suggestions
                        WHERE classification = 'HIGH_PERFORMING'
                        {}
                        ORDER BY analyzed_at DESC, current_metric DESC NULLS LAST
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

                elif intent == "specific_content":
                    sql = """
                        SELECT platform, content_id, title, content_type, canonical_url,
                               classification, current_metric, baseline_metric,
                               likes, comments, peer_explanation, evidence_reason,
                               ai_recommendation, structured_analysis, llm_status, analyzed_at
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
                    # platform_overview / default
                    sql = """
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id)
                                platform, content_id, title, content_type, canonical_url,
                                classification, current_metric, baseline_metric,
                                likes, comments, peer_explanation, evidence_reason,
                                ai_recommendation, analyzed_at
                            FROM ai_suggestions
                            WHERE classification IN ('HIGH_PERFORMING', 'LOW_PERFORMING')
                            {}
                            ORDER BY platform, content_id, analyzed_at DESC
                        )
                        SELECT * FROM latest
                        ORDER BY
                            CASE classification
                                WHEN 'HIGH_PERFORMING' THEN 1
                                WHEN 'LOW_PERFORMING' THEN 2
                                ELSE 3
                            END,
                            current_metric DESC NULLS LAST
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
        "You are a friendly, knowledgeable AI assistant for a media analytics dashboard. "
        "You help the team understand how their content is performing on YouTube, Instagram, and Facebook.\n\n"
        "HOW TO RESPOND:\n"
        "- Be conversational and direct. Answer the question first, then give supporting detail.\n"
        "- For greetings like 'hi' or 'hello', just say hello back naturally. Do not launch into a data report.\n"
        "- For 'thanks' or 'goodbye', respond warmly and briefly.\n"
        "- For analytics questions, give a clear direct answer using the verified data provided.\n"
        "- For follow-up questions, use the conversation history to understand context.\n\n"
        "WRITING STYLE:\n"
        "- Write in plain, natural sentences. No bullet-point overload.\n"
        "- Do not start every answer with 'Based on verified data...' — just answer.\n"
        "- Do not use ** markdown bold or * bullet symbols anywhere in your response.\n"
        "- Do not mention 'records_used', 'provider_status', 'llm_used', 'Source: ai_suggestions', or internal field names.\n"
        "- Do not add a disclaimer on every single message — only mention data limitations when genuinely relevant.\n"
        "- If you do not have enough data to answer, say so simply: "
        "'I don't have enough data for that comparison right now. Try running a fresh scan.'\n\n"
        "WHEN DISCUSSING ANALYTICS:\n"
        "- Give specific titles and numbers when available.\n"
        "- Mention timeframe when relevant.\n"
        "- Give a recommendation when the user asks what to do.\n"
        "- Do not claim a cause unless the data actually supports it.\n"
        "- Distinguish between what the data shows and what might explain it.\n\n"
        "EXAMPLES OF GOOD RESPONSES:\n"
        "User: hi\n"
        "You: Hi! How can I help you?\n\n"
        "User: which videos are performing best?\n"
        "You: The strongest-performing videos right now are [list from data]. "
        "[Title 1] has [X] views and is the top performer in the last 30 days.\n\n"
        "User: why is the first one doing well?\n"
        "You: It's performing strongly because [reason from evidence]. "
        "The engagement rate is also above average, which suggests the audience is responding well to the content.\n\n"
        "User: what should we do next?\n"
        "You: Based on the current data, the strongest move would be to [specific recommendation].\n"
    )


def _build_chat_user_prompt(
    question: str,
    evidence: dict,
    conversation_history: Optional[list[dict]],
) -> str:
    intent = evidence["intent"]
    records = evidence["records"]
    extra = evidence.get("extra", {})

    # For casual conversation intents, no evidence context is needed
    if intent == "casual":
        lines = []
        if conversation_history:
            bounded = conversation_history[-4:]
            lines.append("=== RECENT CONVERSATION ===")
            for turn in bounded:
                role = turn.get("role", "")
                text = (turn.get("content") or turn.get("text") or "")[:300]
                if role and text:
                    lines.append(f"{role.upper()}: {text}")
            lines.append("")
        lines.append(f"USER MESSAGE: {question}")
        lines.append("Respond naturally and conversationally.")
        return "\n".join(lines)

    lines = ["=== DASHBOARD DATA ==="]

    if not records:
        lines.append("No matching records found in the current AI Insights dataset.")
        lines.append("Tell the user there is no data available yet and suggest running a fresh scan.")
    else:
        for i, r in enumerate(records[:8], 1):
            title = r.get('title') or r.get('content_id', 'Unknown')
            platform = r.get('platform', '?')
            ctype = r.get('content_type', 'Video')
            classification = r.get('classification', '?')
            lines.append(f"\n{i}. {title} ({platform} {ctype})")
            lines.append(f"   Performance: {classification}")
            if r.get("canonical_url"):
                lines.append(f"   Link: {r['canonical_url']}")
            if r.get("report_period"):
                lines.append(f"   Period: {r['report_period']}")
            if r.get("current_metric") is not None:
                m_label = r.get("metric_name") or "Views/Reach"
                lines.append(f"   {m_label.capitalize()}: {r['current_metric']:,}")
            if r.get("likes") is not None:
                lines.append(f"   Likes: {r['likes']:,}")
            if r.get("comments") is not None:
                lines.append(f"   Comments: {r['comments']:,}")
            if r.get("peer_explanation"):
                lines.append(f"   Context: {r['peer_explanation']}")
            if r.get("evidence_reason") and r.get("evidence_reason") != r.get("peer_explanation"):
                lines.append(f"   Evidence: {r['evidence_reason']}")
            if r.get("ai_recommendation") and r["ai_recommendation"] not in (
                "Recommendation unavailable",
                "Recommendation unavailable (validation failed)",
                "AI insight generation unavailable",
                "AI insight generation failed",
            ):
                lines.append(f"   AI Insight: {r['ai_recommendation'][:200]}")
            if r.get("analyzed_at"):
                lines.append(f"   Last analyzed: {r['analyzed_at']}")

    if extra.get("publishing_windows"):
        lines.append("\n=== OBSERVED PUBLISHING WINDOWS ===")
        for pw in extra["publishing_windows"]:
            lines.append(f"Platform: {pw['platform']}")
            for h in pw.get("top_hours", []):
                lines.append(
                    f"  {h['hour']:02d}:00 — {h['pct_above_avg']:+.0f}% above average ({h['n_obs']} observations)"
                )
        lines.append("Note: These are observed patterns from historical data, not predictions.")

    lines.append(f"\n=== QUESTION ===\n{question}")

    if conversation_history:
        bounded = conversation_history[-4:]
        lines.append("\n=== RECENT CONVERSATION ===")
        for turn in bounded:
            role = turn.get("role", "")
            text = (turn.get("content") or turn.get("text") or "")[:300]
            if role and text:
                lines.append(f"{role.upper()}: {text}")

    lines.append(
        "\nAnswer the question using only the data above. "
        "Write in plain, natural language. "
        "Do not use ** markdown or bullet symbols. "
        "Do not mention internal field names or database terminology."
    )
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

        # ── 2. Retrieve verified evidence (skip for casual conversation) ──
        if intent == "casual":
            evidence = {
                "intent": "casual",
                "records": [],
                "extra": {},
                "record_count": 0,
            }
        else:
            evidence = await _retrieve_evidence(
                intent,
                body.platform,
                question_safe,
                period=body.period,
                content_type=body.content_type,
            )

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
                    "AI analysis is temporarily unavailable. "
                    "The information shown is from verified dashboard records."
                ) if intent != "casual" else "",
            }

        # For casual conversation, do not add the data disclaimer
        disclaimer = ""
        if intent != "casual":
            disclaimer = "Answer based on verified data from the last AI scan."

        return {
            "ok": True,
            "question": question,
            "answer": llm_answer,
            "intent": intent,
            "records_used": evidence["record_count"],
            "provider_status": "ok",
            "provider": provider_name,
            "llm_used": True,
            "disclaimer": disclaimer,
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
    Uses real retrieved data — not a canned template.
    No markdown bullets or bold text.
    """
    intent = evidence["intent"]
    records = evidence["records"]
    extra = evidence.get("extra", {})

    if intent == "casual":
        return "Hi! I'm here to help with your content performance questions."

    if not records:
        return (
            "I don't have any data matching that question right now. "
            "Try running a fresh AI scan to update the insights."
        )

    lines = ["Here is what the dashboard data shows:"]

    for r in records[:6]:
        title = r.get("title") or r.get("content_id", "Unknown")
        cls = r.get("classification", "")
        platform = r.get("platform", "")
        ctype = r.get("content_type", "Video")
        metric_val = r.get("current_metric")
        metric_str = f" with {metric_val:,} views" if metric_val is not None else ""

        cls_desc: dict[str, str] = {
            "HIGH_PERFORMING": "top performer in the selected period",
            "LOW_PERFORMING": "underperforming relative to peers in the selected period",
        }
        desc = cls_desc.get(cls, cls.lower().replace("_", " "))
        lines.append(f"\n{title} ({platform} {ctype}){metric_str} — {desc}.")

    if extra.get("publishing_windows"):
        lines.append("\nObserved publishing windows:")
        for pw in extra["publishing_windows"]:
            hours_str = ", ".join(
                f"{h['hour']:02d}:00 (+{h['pct_above_avg']:.0f}% above average)"
                for h in pw.get("top_hours", [])
            )
            if hours_str:
                lines.append(f"{pw['platform']}: {hours_str}")
        lines.append("These are patterns observed from historical data, not predictions.")

    lines.append(
        "\nNote: AI analysis is temporarily unavailable. "
        "The figures above are from the last completed scan."
    )
    return "\n".join(lines)

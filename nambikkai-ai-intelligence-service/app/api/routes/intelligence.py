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
from typing import Any, LiteralString, Optional, cast

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.data_sources.postgres import _PLATFORM_CFG, _get_pool
from app.api.routes.suggestions import get_period_days

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/intelligence", tags=["intelligence"])

VALID_PLATFORMS = list(_PLATFORM_CFG.keys())  # ["youtube", "instagram", "facebook"]


def _build_period_condition(period: Optional[str]) -> tuple[Optional[str], list[Any]]:
    """Build SQL condition for period filtering in intelligence queries.

    For known period labels (7d/30d/90d) enforces BOTH:
      - report_period = canonical period label
      - published_at within the rolling window
    This prevents records from one period appearing in queries for another period.
    """
    from app.api.routes.suggestions import _get_canonical_period_label
    if not period:
        return None, []
    canonical = _get_canonical_period_label(period)
    days = get_period_days(period)
    if canonical is not None and days is not None:
        return (
            "LOWER(report_period) = %s AND published_at IS NOT NULL"
            " AND published_at >= NOW() - (CAST(%s AS text) || ' days')::INTERVAL"
            " AND published_at <= NOW()",
            [canonical, str(days)]
        )
    return "LOWER(report_period) = %s", [period.lower().strip()]





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
    # NOTE: The following fields are accepted for backwards-compatibility only.
    # They are intentionally IGNORED by the evidence retrieval pipeline.
    # Chat scope is derived solely from: (1) current question text,
    # (2) conversation_history, (3) independent defaults (30d, all accounts).
    # Dashboard UI filter values must NEVER be forwarded here.
    platform: Optional[str] = None
    period: Optional[str] = None
    content_type: Optional[str] = None
    account_key: Optional[str] = None
    # Bounded conversation history for follow-up context.
    # Client sends the last N turns; server never stores session state.
    # Max 6 turns (3 user + 3 assistant) to keep context bounded.
    conversation_history: Optional[list[dict]] = None


# ── Intent routing ─────────────────────────────────────────────────────────
# Maps the question to the right evidence retrieval query.
# Intent classification determines WHICH evidence to retrieve — NOT a canned answer.

def _classify_intent(question: str, conversation_history: Optional[list[dict]] = None) -> str:
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

    # Specific content follow-ups ("its views", "how many views did it get?", "what about it")
    if any(w in q for w in [
        "this video", "this content", "this post", "this reel", "why is",
        "its views", "its reach", "its likes", "it get", "did it get",
        "what were its", "what was its", "tell me more", "what about it",
        "how many views", "how many likes", "how many comments",
    ]):
        return "specific_content"

    # Publishing time queries (checked before 'best' in high_performing)
    if any(w in q for w in ["publish", "when should", "best time", "post time", "schedule", "timing", "when to post"]):
        return "publishing"

    # High-performing content queries — includes follow-up "what about the highest"
    if any(w in q for w in [
        "high", "surge", "surging", "trending", "viral", "boom", "spike",
        "top", "best", "strongest", "leading", "highest",
    ]):
        return "high_performing"

    # Low-performing content queries — includes follow-up "what about the lowest"
    if any(w in q for w in [
        "low", "underperform", "poor", "struggle", "stalled", "dropping",
        "attention", "improve", "fix", "needs", "worst", "weakest", "lowest",
    ]):
        return "low_performing"

    # Follow-up intent inheritance:
    # Short account-correction or scope-switch messages (e.g. "in aanmegam", "what about media",
    # "for media", "not media") inherit intent from previous user turn.
    short_q = len(q.split()) <= 8
    is_scope_switch = any(kw in q for kw in [
        "what about", "in aanm", "for media", "aanmegam", "aanmeegam", "media",
        "not media", "i told", "i said", "for short", "for video", "for reel",
    ])
    if (short_q or is_scope_switch) and conversation_history:
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                prev_text = (turn.get("content") or turn.get("text") or "").strip()
                if prev_text:
                    prev_intent = _classify_intent(prev_text, None)
                    if prev_intent in ("high_performing", "low_performing", "publishing", "specific_content"):
                        return prev_intent

    return "platform_overview"


def _norm_token(t: str) -> str:
    """Normalize transliterated Tamil and English words phonetically."""
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t.lower())
    t = re.sub(r"aa+", "a", t)
    t = re.sub(r"ee+|ea|ei|e", "i", t)
    t = re.sub(r"oo+|oa", "o", t)
    t = re.sub(r"([b-df-hj-np-tv-z])+", r"", t)
    return " ".join(t.split())


async def _get_canonical_metadata() -> tuple[list[str], list[dict]]:
    """
    Fetch authoritative canonical accounts and content types directly from the database.
    No hardcoded accounts, keys, or types.
    """
    pool = _get_pool()
    accounts: list[str] = []
    content_types: list[dict] = []
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT DISTINCT account_key FROM ai_suggestions WHERE account_key IS NOT NULL")
                rows = await cur.fetchall()
                accounts = [r[0] for r in rows if r[0]]

                await cur.execute(
                    "SELECT DISTINCT platform, content_type FROM ai_suggestions WHERE content_type IS NOT NULL"
                )
                type_rows = await cur.fetchall()
                content_types = [{"platform": r[0], "content_type": r[1]} for r in type_rows if r[0] and r[1]]
    except Exception as exc:
        logger.warning("[_get_canonical_metadata] Error querying DB: %s", exc)

    return accounts, content_types


def _resolve_account_from_question(
    question: str,
    conversation_history: Optional[list[dict]],
    canonical_accounts: list[str],
) -> tuple[Optional[str], bool]:
    """
    Dynamically resolve account without hardcoded names.
    Returns (resolved_account_key, unrecognized_flag).
    """
    if not canonical_accounts:
        return None, False

    q_lower = question.lower()
    q_norm = _norm_token(question)
    q_tokens = q_norm.split()

    # Detect negations like "not media", "not in nambikkai media"
    negated_accounts = set()
    for acct in canonical_accounts:
        acct_norm = _norm_token(acct)
        acct_tokens = acct_norm.split()
        for token in acct_tokens:
            if token in ["nambikai"]:  # skip generic brand prefix
                continue
            if f"not {token}" in q_norm or f"not in {token}" in q_norm:
                negated_accounts.add(acct)

    # Score accounts against current question
    scores: dict[str, int] = {acct: 0 for acct in canonical_accounts}
    for acct in canonical_accounts:
        if acct in negated_accounts:
            scores[acct] = -999
            continue

        acct_norm = _norm_token(acct)
        acct_tokens = [tok for tok in acct_norm.split() if tok != "nambikai"]

        # Check full key
        if acct in q_lower or acct_norm in q_norm:
            scores[acct] += 20

        # Check distinctive tokens
        for tok in acct_tokens:
            if tok in q_tokens or tok in q_norm:
                scores[acct] += 10

    best_acct = max(scores, key=lambda k: scores[k])
    if scores[best_acct] > 0:
        return best_acct, False

    # Check if user mentioned an unrecognized account explicitly
    # e.g., "in zee tamil", "about unknownchannel", "account xyz", "channel abc"
    acct_pattern = re.search(r"\b(?:in|for|on|about|account|channel)\s+([a-zA-Z0-9_\-\s]{3,25})", q_lower)
    if acct_pattern:
        phrase = acct_pattern.group(1).strip()
        phrase_norm = _norm_token(phrase)
        # Check raw lowercase phrase words too — normalization may distort common words
        raw_words = set(phrase.lower().split())
        metric_or_content_words = {
            "youtube", "instagram", "facebook", "video", "short", "reel", "post",
            "30", "7", "90", "days", "views", "reach", "likes", "comments",
            "it", "its", "this", "that", "the", "a", "an",
            "performance", "performing", "performer", "metric", "metrics", "stats",
            "highest", "lowest", "best", "worst", "top", "bottom",
            "high", "low", "surge", "surging", "one", "ones",
        }
        norm_words = set(phrase_norm.split())
        if (
            not norm_words.intersection(metric_or_content_words)
            and not raw_words.intersection(metric_or_content_words)
            and len(phrase_norm) > 2
        ):
            if all(scores[a] <= 0 for a in canonical_accounts):
                return None, True

    # If not in current question, inherit from conversation history (most recent user turn first)
    if conversation_history:
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                prev_text = turn.get("content") or turn.get("text") or ""
                prev_acct, _ = _resolve_account_from_question(prev_text, None, canonical_accounts)
                if prev_acct and prev_acct not in negated_accounts:
                    return prev_acct, False

    return None, False


def _resolve_content_type_from_question(
    question: str,
    conversation_history: Optional[list[dict]],
    canonical_types: list[dict],
    platform: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve content type and implied platform from canonical types in DB.
    Preserves distinctions (YouTube Video vs YouTube Short, Instagram Reel vs Post).
    Returns (content_type, implied_platform).
    """
    q_lower = question.lower()

    if "short" in q_lower or "shorts" in q_lower:
        for ct in canonical_types:
            if "short" in ct["content_type"].lower():
                return ct["content_type"], ct["platform"]
        return "YouTube Short", "youtube"

    if "reel" in q_lower or "reels" in q_lower:
        for ct in canonical_types:
            if "reel" in ct["content_type"].lower():
                return ct["content_type"], ct["platform"]
        return "Instagram Reel", "instagram"

    if "post" in q_lower or "posts" in q_lower:
        for ct in canonical_types:
            if "post" in ct["content_type"].lower():
                return ct["content_type"], ct["platform"]
        return "Post", None

    if "video" in q_lower or "videos" in q_lower:
        if "instagram" in q_lower or "insta" in q_lower or platform == "instagram":
            for ct in canonical_types:
                if "reel" in ct["content_type"].lower():
                    return ct["content_type"], "instagram"
            return "Instagram Reel", "instagram"
        # User explicitly requested Video (not Short, not Reel)
        # Map to canonical YouTube Video (or platform-specific Video)
        for ct in canonical_types:
            if "video" in ct["content_type"].lower() and "short" not in ct["content_type"].lower():
                if not platform or ct["platform"] == platform:
                    return ct["content_type"], ct["platform"]
        return "YouTube Video", "youtube"

    # If not in current question, check conversation history
    if conversation_history:
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                prev_text = turn.get("content") or turn.get("text") or ""
                res_ct, res_pl = _resolve_content_type_from_question(prev_text, None, canonical_types, platform)
                if res_ct:
                    return res_ct, res_pl

    return None, None


def _resolve_platform_from_question(
    question: str,
    conversation_history: Optional[list[dict]],
    implied_platform: Optional[str],
) -> Optional[str]:
    """Resolve platform from explicit user question, implied content type, or history."""
    q_lower = question.lower()
    if "youtube" in q_lower:
        return "youtube"
    if "instagram" in q_lower or "insta" in q_lower:
        return "instagram"
    if "facebook" in q_lower or "fb" in q_lower:
        return "facebook"

    if implied_platform:
        return implied_platform

    # Inherit from conversation history
    if conversation_history:
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                prev_text = (turn.get("content") or turn.get("text") or "").lower()
                if "youtube" in prev_text:
                    return "youtube"
                if "instagram" in prev_text or "insta" in prev_text:
                    return "instagram"
                if "facebook" in prev_text or "fb" in prev_text:
                    return "facebook"

    return None


def _resolve_period_from_question(
    question: str,
    conversation_history: Optional[list[dict]],
) -> str:
    """Resolve period from user question, history, or default 30d."""
    q_lower = question.lower()
    if "7d" in q_lower or "7 day" in q_lower or "seven day" in q_lower:
        return "7d"
    if "90d" in q_lower or "90 day" in q_lower or "three month" in q_lower:
        return "90d"
    if "30d" in q_lower or "30 day" in q_lower or "one month" in q_lower or "month" in q_lower:
        return "30d"

    if conversation_history:
        for turn in reversed(conversation_history):
            if turn.get("role") == "user":
                prev_text = (turn.get("content") or turn.get("text") or "").lower()
                if "7d" in prev_text or "7 day" in prev_text:
                    return "7d"
                if "90d" in prev_text or "90 day" in prev_text:
                    return "90d"
                if "30d" in prev_text or "30 day" in prev_text or "month" in prev_text:
                    return "30d"

    return "30d"


async def _retrieve_evidence(
    intent: str,
    question: str,
    conversation_history: Optional[list[dict]] = None,
    # Optional parameters kept for backwards compatibility but not used as UI overrides:
    platform: Optional[str] = None,
    period: Optional[str] = None,
    content_type: Optional[str] = None,
    account_key: Optional[str] = None,
) -> dict:
    """
    Retrieve verified evidence from the database for the classified intent.
    Uses the exact authoritative ranking and deduplication logic as the Insight Cards.

    The AI Assistant derives its scope EXCLUSIVELY from:
    1. Explicit tokens in the current user message.
    2. Prior conversation context in the current chat turn history.
    3. Independent defaults (30d period, all accounts if not specified).
    Dashboard UI filters are NEVER inherited by the Assistant.

    Returns a dict with:
        - intent: the classified intent
        - records: list of verified DB rows
        - extra: metadata on filters and resolution
        - record_count: number of records retrieved
    """
    pool = _get_pool()
    valid_platforms = list(_PLATFORM_CFG.keys())

    # Fetch canonical metadata from database
    canonical_accounts, canonical_types = await _get_canonical_metadata()

    # 1. Resolve Account purely from question and conversation history
    acct_filter, unrecognized_account = _resolve_account_from_question(
        question, conversation_history, canonical_accounts
    )

    # 2. Resolve Content Type and implied Platform purely from question and history
    inferred_content_type, implied_plat = _resolve_content_type_from_question(
        question, conversation_history, canonical_types, None
    )

    # 3. Resolve Platform purely from question, implied content type, or history
    plat_filter = _resolve_platform_from_question(question, conversation_history, implied_plat)

    # 4. Resolve Period purely from question, history, or independent default (30d)
    inferred_period = _resolve_period_from_question(question, conversation_history)
    if not inferred_period:
        inferred_period = "30d"

    # Friendly display name for account
    account_display = "All Accounts"
    if acct_filter:
        account_display = acct_filter.replace("-", " ").replace("_", " ").title()
        if "Anmigam" in account_display or "Aanmigam" in account_display:
            account_display = "Nambikkai Aanmeegam"
        elif "Media" in account_display:
            account_display = "Nambikkai Media"

    extra: dict = {
        "platform_filter": plat_filter,
        "content_type_filter": inferred_content_type,
        "period_filter": inferred_period,
        "account_filter": acct_filter,
        "account_display": account_display,
        "unrecognized_account": unrecognized_account,
    }

    records: list[dict] = []

    # If the user explicitly requested an account that does not exist in DB:
    # Do NOT substitute another account! Return empty records with unrecognized flag.
    if unrecognized_account:
        return {
            "intent": intent,
            "records": [],
            "extra": extra,
            "record_count": 0,
        }

    # Build shared base WHERE conditions
    base_conditions: list[str] = []
    base_params: list[Any] = []

    if plat_filter:
        base_conditions.append("platform = %s")
        base_params.append(plat_filter)

    if acct_filter:
        base_conditions.append("account_key = %s")
        base_params.append(acct_filter)

    if inferred_content_type:
        if "%" in inferred_content_type:
            base_conditions.append("content_type ILIKE %s")
            base_params.append(inferred_content_type)
        else:
            base_conditions.append("(content_type = %s OR content_type ILIKE %s)")
            base_params.append(inferred_content_type)
            base_params.append(f"%{inferred_content_type}%")

    p_cond, p_params = _build_period_condition(inferred_period)
    if p_cond:
        base_conditions.append(p_cond)
        base_params.extend(p_params)

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:

                if intent == "high_performing":
                    clauses = ["classification = 'HIGH_PERFORMING'"] + base_conditions
                    where_str = f"WHERE {' AND '.join(clauses)}"
                    sql = f"""
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id)
                                id, platform, content_id, account_key, title, content_type, canonical_url,
                                classification, current_metric, baseline_metric, metric_name,
                                likes, comments, peer_explanation, evidence_reason,
                                ai_recommendation, report_period, analyzed_at
                            FROM ai_suggestions
                            {where_str}
                            ORDER BY platform, content_id, analyzed_at DESC
                        )
                        SELECT * FROM latest
                        ORDER BY current_metric DESC NULLS LAST, analyzed_at DESC
                        LIMIT 8
                    """
                    await cur.execute(cast(LiteralString, sql), base_params)

                elif intent == "low_performing":
                    clauses = ["classification = 'LOW_PERFORMING'"] + base_conditions
                    where_str = f"WHERE {' AND '.join(clauses)}"
                    sql = f"""
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id)
                                id, platform, content_id, account_key, title, content_type, canonical_url,
                                classification, current_metric, baseline_metric, metric_name,
                                likes, comments, peer_explanation, evidence_reason,
                                ai_recommendation, report_period, analyzed_at
                            FROM ai_suggestions
                            {where_str}
                            ORDER BY platform, content_id, analyzed_at DESC
                        )
                        SELECT * FROM latest
                        ORDER BY current_metric ASC NULLS LAST, analyzed_at DESC
                        LIMIT 8
                    """
                    await cur.execute(cast(LiteralString, sql), base_params)

                elif intent == "publishing":
                    pub_conditions = ["classification = 'HIGH_PERFORMING'"]
                    pub_params = []
                    if plat_filter:
                        pub_conditions.append("platform = %s")
                        pub_params.append(plat_filter)
                    if acct_filter:
                        pub_conditions.append("account_key = %s")
                        pub_params.append(acct_filter)
                    if inferred_content_type:
                        pub_conditions.append("content_type ILIKE %s")
                        pub_params.append(f"%{inferred_content_type.replace('%', '')}%")
                    p_cond, p_params = _build_period_condition(inferred_period)
                    if p_cond:
                        pub_conditions.append(p_cond)
                        pub_params.extend(p_params)
                    where_str = f"WHERE {' AND '.join(pub_conditions)}"
                    sql = f"""
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id)
                                id, platform, content_id, account_key, title, content_type, canonical_url,
                                classification, current_metric, baseline_metric, metric_name,
                                likes, comments, peer_explanation, evidence_reason,
                                ai_recommendation, report_period, analyzed_at
                            FROM ai_suggestions
                            {where_str}
                            ORDER BY platform, content_id, analyzed_at DESC
                        )
                        SELECT * FROM latest
                        ORDER BY current_metric DESC NULLS LAST, analyzed_at DESC
                        LIMIT 5
                    """
                    await cur.execute(cast(LiteralString, sql), pub_params)

                    # Also fetch publishing hour intelligence for context
                    platforms_to_check = [plat_filter] if plat_filter else valid_platforms
                    pub_windows: list[dict] = []
                    for p in platforms_to_check[:2]:
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
                    spec_conditions = []
                    spec_params = []
                    if plat_filter:
                        spec_conditions.append("platform = %s")
                        spec_params.append(plat_filter)
                    if acct_filter:
                        spec_conditions.append("account_key = %s")
                        spec_params.append(acct_filter)
                    if inferred_content_type:
                        spec_conditions.append("content_type ILIKE %s")
                        spec_params.append(f"%{inferred_content_type.replace('%', '')}%")
                    p_cond, p_params = _build_period_condition(inferred_period)
                    if p_cond:
                        spec_conditions.append(p_cond)
                        spec_params.extend(p_params)
                    where_str = f"WHERE {' AND '.join(spec_conditions)}" if spec_conditions else ""
                    sql = f"""
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id)
                                id, platform, content_id, account_key, title, content_type, canonical_url,
                                classification, current_metric, baseline_metric, metric_name,
                                likes, comments, peer_explanation, evidence_reason,
                                ai_recommendation, structured_analysis, llm_status, report_period, analyzed_at
                            FROM ai_suggestions
                            {where_str}
                            ORDER BY platform, content_id, analyzed_at DESC
                        )
                        SELECT * FROM latest
                        ORDER BY analyzed_at DESC
                        LIMIT 5
                    """
                    await cur.execute(cast(LiteralString, sql), spec_params)

                else:
                    # platform_overview / default
                    clauses = ["classification IN ('HIGH_PERFORMING', 'LOW_PERFORMING')"] + base_conditions
                    where_str = f"WHERE {' AND '.join(clauses)}"
                    sql = f"""
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id)
                                id, platform, content_id, account_key, title, content_type, canonical_url,
                                classification, current_metric, baseline_metric, metric_name,
                                likes, comments, peer_explanation, evidence_reason,
                                ai_recommendation, report_period, analyzed_at
                            FROM ai_suggestions
                            {where_str}
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
                            analyzed_at DESC
                        LIMIT 10
                    """
                    await cur.execute(cast(LiteralString, sql), base_params)

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

                # Fallback for YouTube: If the user asked about 'video' generally on YouTube,
                # but all high performers in that period are YouTube Shorts, retrieve YouTube content
                # so the user gets the true top performer instead of a false 'no data' answer.
                if (
                    not records
                    and plat_filter == "youtube"
                    and inferred_content_type == "YouTube Video"
                    and "regular" not in question.lower()
                    and "long" not in question.lower()
                ):
                    intent_cls = "HIGH_PERFORMING" if intent == "high_performing" else "LOW_PERFORMING"
                    yt_clauses = [f"classification = '{intent_cls}'"]
                    yt_params = []
                    yt_clauses.append("platform = %s")
                    yt_params.append(plat_filter)
                    if acct_filter:
                        yt_clauses.append("account_key = %s")
                        yt_params.append(acct_filter)
                    p_cond, p_params = _build_period_condition(inferred_period)
                    if p_cond:
                        yt_clauses.append(p_cond)
                        yt_params.extend(p_params)
                    yt_where = f"WHERE {' AND '.join(yt_clauses)}"
                    yt_order = "current_metric DESC NULLS LAST" if intent == "high_performing" else "current_metric ASC NULLS LAST"
                    yt_sql = f"""
                        WITH latest AS (
                            SELECT DISTINCT ON (platform, content_id)
                                id, platform, content_id, account_key, title, content_type, canonical_url,
                                classification, current_metric, baseline_metric, metric_name,
                                likes, comments, peer_explanation, evidence_reason,
                                ai_recommendation, report_period, analyzed_at
                            FROM ai_suggestions
                            {yt_where}
                            ORDER BY platform, content_id, analyzed_at DESC
                        )
                        SELECT * FROM latest
                        ORDER BY {yt_order}, analyzed_at DESC
                        LIMIT 8
                    """
                    await cur.execute(cast(LiteralString, yt_sql), yt_params)
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
                    if records:
                        extra["content_type_note"] = (
                            f"Top YouTube performer in this period is a {records[0].get('content_type')} "
                            f"(no long-form YouTube Video reached the high-performing threshold)."
                        )

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
        "You are a friendly, grounded AI assistant for a media analytics dashboard. "
        "You provide reliable performance analysis across YouTube, Instagram, and Facebook based strictly on verified dashboard data.\n\n"
        "CORE RULES:\n"
        "- Answer directly first. Be concise and natural.\n"
        "- For greetings like 'hi' or 'hello', just greet warmly. Do not output analytics reports for greetings.\n"
        "- For 'thanks' or 'bye', respond warmly and briefly.\n"
        "- When asked for highest or lowest performing content, the records in the evidence are pre-ranked by the dashboard performance engine. Report Record #1 as the primary authoritative answer. Never select a different record or invent your own ranking formula.\n"
        "- When referring to accounts, use their clean display names (e.g. Nambikkai Aanmeegam or Nambikkai Media). Never expose raw internal database keys like 'nambikkai-anmigam'.\n"
        "- If the requested account or content type has no verified records in the selected period, state clearly that no verified data is available for that specific scope. NEVER substitute a different account or content type silently.\n\n"
        "GROUNDED ANALYTICS & NO FABRICATED METRICS:\n"
        "- State ONLY metrics and facts present in the verified record (Views, Reach, Likes, Comments, Classification).\n"
        "- NEVER invent or calculate claims like '100% below average', '99% below average', 'below similar videos', or comment rate benchmarks unless explicitly provided in the verified data.\n"
        "- If asked about peer comparisons, benchmarks, or why content performed a certain way, state honestly that you can confirm the verified metrics and ranking, but do not have a verified peer-average benchmark or causal attribution.\n\n"
        "FORMATTING — PLAIN TEXT ONLY:\n"
        "- Use PLAIN TEXT. No markdown. No ** ** bold markers. No * * italic. No __ __ underline. No # headings. No ``` code blocks.\n"
        "- Use a hyphen-space (- ) for bullet points. Never use asterisks for bullets.\n"
        "- Tamil titles must be preserved exactly as given in the evidence, without modification.\n"
        "- Never mention 'records_used', 'provider_status', 'llm_used', SQL queries, or internal database field names.\n"
        "- Never output literal backslash-asterisk or escaped characters.\n"
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

    lines = ["=== VERIFIED DASHBOARD EVIDENCE ==="]
    scope_parts = []
    if extra.get("account_display"):
        scope_parts.append(f"Account: {extra['account_display']}")
    elif extra.get("account_filter"):
        scope_parts.append(f"Account: {extra['account_filter']}")
    if extra.get("platform_filter"):
        scope_parts.append(f"Platform: {extra['platform_filter'].capitalize()}")
    if extra.get("content_type_filter"):
        scope_parts.append(f"Content Type: {extra['content_type_filter']}")
    if extra.get("period_filter"):
        scope_parts.append(f"Period: {extra['period_filter'].upper()}")
    if scope_parts:
        lines.append(f"Active Scope: {', '.join(scope_parts)}")
    if extra.get("content_type_note"):
        lines.append(f"Scope Note: {extra['content_type_note']}")

    if extra.get("unrecognized_account"):
        lines.append("\nNOTICE: The account requested by the user was not found in the verified dashboard accounts.")
        lines.append("Explicitly inform the user that the requested account was not found. Do NOT substitute any other account.")
    elif not records:
        scope_desc = []
        if extra.get("account_display"):
            scope_desc.append(f"in {extra['account_display']}")
        if extra.get("content_type_filter"):
            scope_desc.append(f"{extra['content_type_filter']}")
        if extra.get("platform_filter"):
            scope_desc.append(f"on {extra['platform_filter'].capitalize()}")
        if extra.get("period_filter"):
            scope_desc.append(f"over the last {extra['period_filter'].upper()}")
        scope_str = " ".join(scope_desc) if scope_desc else "for the requested criteria"
        lines.append(f"\nNOTICE: No verified records found {scope_str} in the dashboard dataset.")
        lines.append("Inform the user that there is no verified data available for this specific scope.")
        lines.append("Do NOT substitute a different account or a different content type.")
    else:
        for i, r in enumerate(records[:8], 1):
            title = r.get("title") or r.get("content_id", "Unknown")
            platform = r.get("platform", "?")
            ctype = r.get("content_type", "Content")
            classification = r.get("classification", "?")
            rank_label = f" (Authoritative #{i} Result)" if i == 1 else f" (#{i})"
            acct_display = extra.get("account_display") or r.get("account_key") or ""
            acct_tag = f" - {acct_display}" if acct_display else ""

            lines.append(f"\n{i}. {title} ({platform.capitalize()} {ctype}{acct_tag}){rank_label}")
            lines.append(f"   Performance Classification: {classification}")
            if r.get("canonical_url"):
                lines.append(f"   Link: {r['canonical_url']}")
            if r.get("report_period"):
                lines.append(f"   Period: {r['report_period'].upper()}")
            if r.get("current_metric") is not None:
                m_label = r.get("metric_name") or ("Views" if platform.lower() == "youtube" else "Reach")
                lines.append(f"   {m_label.capitalize()}: {r['current_metric']:,}")
            if r.get("likes") is not None:
                lines.append(f"   Likes: {r['likes']:,}")
            if r.get("comments") is not None:
                lines.append(f"   Comments: {r['comments']:,}")

            # Strip legacy unverified percentage comparisons
            peer_exp = r.get("peer_explanation") or ""
            ev_reason = r.get("evidence_reason") or ""
            if peer_exp and "%" not in peer_exp and "below average" not in peer_exp.lower():
                lines.append(f"   Context: {peer_exp}")
            elif ev_reason and "%" not in ev_reason and "below average" not in ev_reason.lower():
                lines.append(f"   Evidence: {ev_reason}")

            if r.get("ai_recommendation") and r["ai_recommendation"] not in (
                "Recommendation unavailable",
                "Recommendation unavailable (validation failed)",
                "AI insight generation unavailable",
                "AI insight generation failed",
            ):
                lines.append(f"   AI Insight: {r['ai_recommendation'][:200]}")

    if extra.get("publishing_windows"):
        lines.append("\n=== OBSERVED PUBLISHING WINDOWS ===")
        for pw in extra["publishing_windows"]:
            lines.append(f"Platform: {pw['platform'].capitalize()}")
            for h in pw.get("top_hours", []):
                lines.append(
                    f"  {h['hour']:02d}:00 — {h['pct_above_avg']:+.0f}% above average ({h['n_obs']} observations)"
                )
        lines.append("Note: Observed historical patterns, not predictions.")

    if intent in ("high_performing", "low_performing") and records:
        direction = "highest" if intent == "high_performing" else "lowest"
        lines.append(
            f"\n=== AUTHORITATIVE RANKING INSTRUCTION ===\n"
            f"The dashboard performance engine has already evaluated and ranked the records above. "
            f"Record #1 is the authoritative #{direction}-performing item according to the dashboard performance engine. "
            f"You MUST identify Record #1 as the primary answer to the user's question. "
            f"Do NOT recalculate ranking or pick a different record. "
            f"Present Record #1 clearly with Title, Platform, Content Type, Views/Reach, Likes, and Classification."
        )

    lines.append(
        "\n=== STRICT ACCURACY INSTRUCTION ===\n"
        "- Do NOT claim '100% below average' or invent percentage comparisons.\n"
        "- If peer average is not in the data, state that verified peer-average benchmark is not available.\n"
        "- Do NOT substitute a different account or content type if data is unavailable."
    )

    lines.append(f"\n=== USER QUESTION ===\n{question}")

    if conversation_history:
        bounded = conversation_history[-4:]
        lines.append("\n=== RECENT CONVERSATION ===")
        for turn in bounded:
            role = turn.get("role", "")
            text = (turn.get("content") or turn.get("text") or "")[:300]
            if role and text:
                lines.append(f"{role.upper()}: {text}")

    lines.append(
        "\nAnswer conversationally and directly. Do not use internal field names or database terms."
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
        intent = _classify_intent(question_safe, history)
        logger.info("[chat] intent=%s platform=%s question_len=%d", intent, body.platform, len(question))

        # ── 2. Retrieve verified evidence (skip for casual conversation) ──
        # Chat scope is derived EXCLUSIVELY from: (1) current user message, (2) conversation
        # history, (3) independent defaults. Dashboard UI filter values (body.platform,
        # body.period, body.account_key, body.content_type) are intentionally NOT passed here.
        evidence: dict[str, Any]
        if intent == "casual":
            evidence = {
                "intent": "casual",
                "records": [],
                "extra": {},
                "record_count": 0,
            }
        else:
            evidence = await _retrieve_evidence(
                intent=intent,
                question=question_safe,
                conversation_history=history,
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

        # ── 5. Strip markdown artifacts from LLM output ──────────────────
        if llm_answer:
            llm_answer = _strip_markdown(llm_answer)

        # ── 6. Validate ranking answer against authoritative record ───────
        # For high/low ranking questions with a primary record, verify the LLM
        # didn't hallucinate a different content item. If the answer omits the
        # primary record's content_id/title, use the safe fallback instead.
        records_list: Any = evidence.get("records")
        if (
            llm_answer
            and intent in ("high_performing", "low_performing")
            and isinstance(records_list, list)
            and len(records_list) > 0
        ):
            primary = records_list[0]
            if isinstance(primary, dict):
                primary_id = str(primary.get("content_id") or "")
                primary_title = str(primary.get("title") or "")[:40]
                # Validate: LLM answer must at minimum reference the primary content_id
                # OR a meaningful fragment of its title. Titles may be in Tamil Unicode.
                id_in_answer = bool(primary_id and primary_id in llm_answer)
                title_fragment = primary_title[:20].strip() if primary_title else ""
                title_in_answer = bool(title_fragment and title_fragment in llm_answer)
                # If LLM omitted both, fall back to verified answer
                if primary_id and not id_in_answer and not title_in_answer:
                    logger.warning(
                        "[chat] LLM answer did not reference authoritative primary record '%s'; using fallback",
                        primary_id,
                    )
                    llm_answer = _build_fallback_answer(evidence)

        # ── 7. Build response ────────────────────────────────────────────
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


def _strip_markdown(text: str) -> str:
    """
    Remove markdown formatting from LLM output so plain-text chat UI shows clean text.

    The chat UI renders messages with whitespace-pre-wrap (no markdown parser), so
    bold markers like **word** display as literal asterisks. This function strips them.
    Preserves Tamil Unicode, hyphens used as bullet points, and plain punctuation.
    """
    # Remove triple bold/italic (***text***)
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text, flags=re.DOTALL)
    # Remove bold (**text**)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text, flags=re.DOTALL)
    # Remove italic (*text*) — but preserve standalone hyphens used as bullet points
    text = re.sub(r'(?<![\-\s])\*(.+?)\*(?![\-\s])', r'\1', text, flags=re.DOTALL)
    # Remove __bold__ and _italic_
    text = re.sub(r'_{2}(.+?)_{2}', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'_(.+?)_', r'\1', text, flags=re.DOTALL)
    # Remove escaped asterisks (\*)
    text = text.replace('\\*', '')
    # Remove ATX headings (# Heading, ## Heading, etc.) at line start
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove backtick inline code (preserve content)
    text = re.sub(r'`(.+?)`', r'\1', text)
    # Collapse any repeated blank lines (more than 2 newlines in a row)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


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
        scope_parts = []
        acct_disp = extra.get("account_display") or extra.get("account_filter") or ""
        if acct_disp:
            scope_parts.append(acct_disp)
        ct = extra.get("content_type_filter") or ""
        if ct:
            scope_parts.append(ct)
        pl = extra.get("platform_filter") or ""
        if pl and not ct:
            scope_parts.append(pl.capitalize())
        pr = extra.get("period_filter") or ""
        if pr:
            scope_parts.append(pr.upper())
        scope_desc = " / ".join(scope_parts) if scope_parts else "the requested criteria"
        return (
            f"No verified data is available for {scope_desc} in the current dataset. "
            "This may mean no content was scanned for this combination yet. "
            "Try running a fresh scan or adjusting the filters."
        )

    lines = ["Here is what the dashboard data shows:"]

    for r in records[:6]:
        title = r.get("title") or r.get("content_id", "Unknown")
        cls = r.get("classification", "")
        platform = r.get("platform", "")
        ctype = r.get("content_type") or "Video"
        metric_val = r.get("current_metric")
        m_label = r.get("metric_name") or ("views" if platform == "youtube" else "reach")
        metric_str = f" - {metric_val:,} {m_label}" if metric_val is not None else ""

        cls_desc: dict[str, str] = {
            "HIGH_PERFORMING": "top performer in the selected period",
            "LOW_PERFORMING": "underperforming relative to peers in the selected period",
        }
        desc = cls_desc.get(cls, cls.lower().replace("_", " "))
        lines.append(f"\n- {title} ({platform.capitalize()} {ctype}){metric_str} - {desc}.")

    if extra.get("publishing_windows"):
        lines.append("\nObserved publishing windows:")
        for pw in extra["publishing_windows"]:
            hours_str = ", ".join(
                f"{h['hour']:02d}:00 (+{h['pct_above_avg']:.0f}% above average)"
                for h in pw.get("top_hours", [])
            )
            if hours_str:
                lines.append(f"- {pw['platform'].capitalize()}: {hours_str}")
        lines.append("These are patterns observed from historical data, not predictions.")

    if extra.get("content_type_note"):
        lines.append(f"\n{extra['content_type_note']}")

    lines.append(
        "\nNote: AI analysis is temporarily unavailable. "
        "The figures above are from the last completed scan."
    )
    return "\n".join(lines)

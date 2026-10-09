"""
Bulk scanner service for AI Performance Insights.

Processes authoritative performance candidates (HIGH_PERFORMING, LOW_PERFORMING)
supplied exclusively by the dashboard performance engine.

Flow:
Dashboard Candidates -> Validate Payload -> Build Bounded Evidence -> LLM Explanation -> ai_suggestions Store

The AI service does NOT:
- Calculate, classify, gate, rank, predict, or reinterpret performance.
- Use XGBoost, LightGBM, velocity ratios, or like acceleration.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, cast
import re
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from app.core.config import get_settings
from app.core.exceptions import ProviderError, ProviderUnavailableError
from app.data_sources.postgres import _get_pool
from app.domain.models import PerformanceCandidate
from app.providers import get_provider_with_fallback
from app.services.evidence_builder import build_evidence_package
from app.validation.output_validator import validate_output
from app.validation.policy_validator import validate_policy

logger = logging.getLogger(__name__)

# Valid visible insight classifications
_VISIBLE_INSIGHT_CLASSIFICATIONS = {"HIGH_PERFORMING", "LOW_PERFORMING"}

_UPSERT_SQL = """
INSERT INTO ai_suggestions (
    platform, content_id, account_key,
    title, content_type, canonical_url, metric_name,
    classification, current_metric, baseline_metric,
    likes, comments, peer_explanation,
    evidence_reason, ai_recommendation,
    structured_analysis, llm_status,
    report_period, is_low_performing, published_at, analyzed_at
) VALUES (
    %(platform)s, %(content_id)s, %(account_key)s,
    %(title)s, %(content_type)s, %(canonical_url)s, %(metric_name)s,
    %(classification)s, %(current_metric)s, %(baseline_metric)s,
    %(likes)s, %(comments)s, %(peer_explanation)s,
    %(evidence_reason)s, %(ai_recommendation)s,
    %(structured_analysis)s, %(llm_status)s,
    %(report_period)s, %(is_low_performing)s, %(published_at)s, NOW()
)
ON CONFLICT (platform, content_id, report_period) DO UPDATE SET
    account_key = EXCLUDED.account_key,
    title = EXCLUDED.title,
    content_type = EXCLUDED.content_type,
    canonical_url = CASE
        WHEN EXCLUDED.canonical_url IS NOT NULL AND EXCLUDED.canonical_url != '' THEN EXCLUDED.canonical_url
        ELSE ai_suggestions.canonical_url
    END,
    metric_name = EXCLUDED.metric_name,
    classification = EXCLUDED.classification,
    current_metric = EXCLUDED.current_metric,
    baseline_metric = EXCLUDED.baseline_metric,
    likes = EXCLUDED.likes,
    comments = EXCLUDED.comments,
    peer_explanation = EXCLUDED.peer_explanation,
    evidence_reason = EXCLUDED.evidence_reason,
    ai_recommendation = EXCLUDED.ai_recommendation,
    structured_analysis = EXCLUDED.structured_analysis,
    llm_status = EXCLUDED.llm_status,
    is_low_performing = EXCLUDED.is_low_performing,
    published_at = COALESCE(EXCLUDED.published_at, ai_suggestions.published_at),
    analyzed_at = NOW();
"""


@dataclass
class ScanSummary:
    platform: str
    total_content_ids: int = 0
    scanned: int = 0
    actionable: int = 0
    errors: int = 0
    suggestions_updated: int = 0
    recommendations_attempted: int = 0
    recommendations_generated: int = 0
    recommendations_reused: int = 0
    recommendations_failed: int = 0
    llm_provider: str = "none"



def clean_canonical_url(url: str, platform: str, content_id: str, content_type: str = "") -> str:
    raw = (url or "").strip()
    is_cdn = any(cdn in raw.lower() for cdn in [
        "fbcdn.net", "cdninstagram.com", "akamaihd.net", "fbsbx.com", "cdn.", ".fbcdn.",
        "googlevideo.com", "ytimg.com"
    ]) or any(raw.lower().endswith(ext) for ext in [".mp4", ".m4v", ".webm", ".jpg", ".jpeg", ".png", ".webp"])

    if is_cdn:
        raw = ""

    p = (platform or "").lower()
    ct = (content_type or "").lower()
    cid = (content_id or "").strip()

    if p == "youtube":
        if raw and ("youtube.com" in raw or "youtu.be" in raw):
            return raw
        if cid:
            return f"https://www.youtube.com/shorts/{cid}" if "short" in ct else f"https://www.youtube.com/watch?v={cid}"
        return ""
    elif p == "instagram":
        # Accept only authentic Instagram permalinks with shortcode
        # NEVER construct /reel/{id} from numeric media_id!
        if raw and "instagram.com" in raw:
            if not re.search(r'/(?:reel|p)/\d{10,}/?', raw):
                return raw
        return ""
    elif p == "facebook":
        if raw and "facebook.com" in raw and not is_cdn:
            return raw
        return ""
    return raw


async def scan_with_candidates(
    candidates: List[dict],
    force_refresh: bool = False,
    period: str = "30d",
) -> List[ScanSummary]:
    """
    Process authoritative performance candidates from the dashboard.

    Parameters
    ----------
    candidates:
        List of candidate dicts from performance-candidates.ts with:
          platform, content_id, account_key, performance_level,
          title, url / canonical_url, content_type, published_at,
          current_metric, metric_name, likes, comments, peer_explanation,
          transcript, transcript_status
    force_refresh:
        When True, re-generates LLM recommendations even if cached.
    period:
        The analysis period: '7d', '30d', or '90d'.
    """
    if not candidates:
        logger.info("[CandidateScan] Received empty candidate list; nothing to scan.")
        return []

    # Group candidates by platform
    by_platform: dict[str, List[dict]] = {}
    for c in candidates:
        plat = (c.get("platform") or "").lower().strip()
        if plat not in ("youtube", "instagram", "facebook"):
            logger.warning("[CandidateScan] Unknown platform '%s', skipping", plat)
            continue
        by_platform.setdefault(plat, []).append(c)

    # Initialize LLM provider
    provider = None
    fallback_provider = None
    provider_name = "unavailable"
    try:
        provider, fallback_provider = get_provider_with_fallback()
        provider_name = provider.get_provider_metadata().get("provider", "unknown")
    except Exception as p_err:
        logger.warning("[CandidateScan] Could not initialize LLM provider: %s", p_err)

    pool = _get_pool()
    summaries: List[ScanSummary] = []

    for platform, plat_candidates in by_platform.items():
        summary = ScanSummary(platform=platform)
        summary.llm_provider = provider_name
        summary.total_content_ids = len(plat_candidates)
        summary.scanned = len(plat_candidates)
        summary.actionable = len(plat_candidates)

        # Load existing generated suggestions for cache logic scoped to this platform and period
        existing_generated: dict[str, dict] = {}
        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        SELECT content_id, ai_recommendation, structured_analysis
                        FROM ai_suggestions
                        WHERE platform = %s AND report_period = %s AND llm_status = 'generated'
                          AND structured_analysis IS NOT NULL
                        """,
                        (platform, period),
                    )
                    for r in await cur.fetchall():
                        existing_generated[r[0]] = {
                            "ai_recommendation": r[1],
                            "structured_analysis": r[2],
                        }
        except Exception as exc:
            logger.warning(
                "[CandidateScan] Could not load existing suggestions for %s (%s): %s",
                platform, period, exc,
            )

        concurrency = get_settings().LLM_CONCURRENCY
        semaphore = asyncio.Semaphore(max(1, concurrency))
        rows_to_upsert: dict[str, dict] = {}

        async def _process_candidate(cand: dict) -> tuple[str, dict]:
            content_id = str(cand.get("content_id") or "").strip()
            perf_level = str(cand.get("performance_level") or "HIGH_PERFORMING").upper()
            if perf_level not in _VISIBLE_INSIGHT_CLASSIFICATIONS:
                perf_level = "HIGH_PERFORMING" if "HIGH" in perf_level else "LOW_PERFORMING"

            title = str(cand.get("title") or f"{platform}/{content_id}")[:200]
            account_key = str(cand.get("account_key") or "")
            content_type = str(cand.get("content_type") or "Video")
            raw_url = str(cand.get("canonical_url") or cand.get("url") or "")
            canonical_url = clean_canonical_url(raw_url, platform=platform, content_id=content_id, content_type=content_type)
            metric_name = str(cand.get("metric_name") or ("views" if platform == "youtube" else "reach"))
            curr_val = int(float(cand.get("current_metric") or 0))
            base_val = float(cand["baseline_metric"]) if cand.get("baseline_metric") is not None else None
            likes_val = int(float(cand.get("likes") or 0))
            comments_val = int(float(cand.get("comments") or 0))
            peer_explanation = str(cand.get("peer_explanation") or "")
            transcript_text = cand.get("transcript") or None

            is_low = perf_level == "LOW_PERFORMING"

            pub_raw = cand.get("published_at")
            pub_dt = None
            if pub_raw:
                if isinstance(pub_raw, datetime):
                    pub_dt = pub_raw
                elif isinstance(pub_raw, str):
                    try:
                        pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                    except Exception:
                        pub_dt = None

            base_row: dict[str, Any] = {
                "platform": platform,
                "content_id": content_id,
                "account_key": account_key,
                "title": title,
                "content_type": content_type,
                "canonical_url": canonical_url,
                "metric_name": metric_name,
                "classification": perf_level,
                "current_metric": curr_val,
                "baseline_metric": base_val,
                "likes": likes_val,
                "comments": comments_val,
                "peer_explanation": peer_explanation,
                "evidence_reason": peer_explanation or f"Authoritative {perf_level} from dashboard",
                "report_period": period,
                "is_low_performing": is_low,
                "published_at": pub_dt,
                "ai_recommendation": None,
                "structured_analysis": cast(Any, None),
                "llm_status": None,
            }

            # Cache hit — reuse previously generated LLM output unless force_refresh requested
            if content_id in existing_generated and not force_refresh:
                cached = existing_generated[content_id]
                base_row["ai_recommendation"] = cached.get("ai_recommendation", "Recommendation unavailable")
                cached_sa = cached.get("structured_analysis")
                base_row["structured_analysis"] = Jsonb(cached_sa) if cached_sa else None
                base_row["llm_status"] = "generated"
                return content_id, base_row

            # LLM provider unavailable
            if provider is None:
                base_row["ai_recommendation"] = "AI recommendation temporarily unavailable (LLM unconfigured or unreachable) — verified performance metrics are intact."
                base_row["structured_analysis"] = None
                base_row["llm_status"] = "unavailable"
                return content_id, base_row

            # Generate LLM insight
            async with semaphore:
                try:
                    candidate_obj = PerformanceCandidate(
                        platform=platform,
                        content_id=content_id,
                        account_key=account_key,
                        performance_level=perf_level,
                        content_type=content_type,
                        title=title,
                        url=canonical_url,
                        canonical_url=canonical_url,
                        published_at=str(pub_raw) if pub_raw else None,
                        period=period,
                        current_metric=float(curr_val),
                        baseline_metric=base_val,
                        metric_name=metric_name,
                        likes=float(likes_val),
                        comments=float(comments_val),
                        peer_explanation=peer_explanation,
                        transcript=transcript_text,
                    )
                    evidence = build_evidence_package(
                        candidate=candidate_obj,
                        transcript_excerpt=transcript_text,
                    )

                    analysis = None
                    try:
                        analysis = await provider.generate_structured_analysis(evidence)
                    except ProviderUnavailableError as primary_exc:
                        if fallback_provider is not None:
                            logger.warning(
                                "[CandidateScan] Primary provider unavailable for %s/%s: %s. Trying fallback.",
                                platform, content_id, primary_exc,
                            )
                            try:
                                analysis = await fallback_provider.generate_structured_analysis(evidence)
                            except Exception as fb_exc:
                                logger.warning(
                                    "[CandidateScan] Fallback provider also failed for %s/%s: %s",
                                    platform, content_id, fb_exc,
                                )
                        else:
                            raise

                    if analysis is not None:
                        out_val = validate_output(analysis, evidence)
                        pol_val = validate_policy(analysis)

                        # Bounded retry if initial validation failed (at most 1 retry with actual feedback)
                        if not out_val.is_valid or not pol_val.is_valid:
                            first_failures = out_val.failures + pol_val.failures
                            feedback = "; ".join(first_failures)
                            logger.info(
                                "[CandidateScan] Initial validation failed for %s/%s (%s). Attempting bounded retry with feedback...",
                                platform, content_id, feedback,
                            )
                            try:
                                retry_analysis = await provider.generate_structured_analysis(evidence, feedback=feedback)
                                retry_out = validate_output(retry_analysis, evidence)
                                retry_pol = validate_policy(retry_analysis)
                                if retry_out.is_valid and retry_pol.is_valid:
                                    logger.info("[CandidateScan] Bounded retry succeeded for %s/%s", platform, content_id)
                                    analysis = retry_analysis
                                    out_val = retry_out
                                    pol_val = retry_pol
                                else:
                                    logger.warning(
                                        "[CandidateScan] Bounded retry also failed for %s/%s: %s",
                                        platform, content_id, "; ".join(retry_out.failures + retry_pol.failures),
                                    )
                            except Exception as retry_exc:
                                logger.warning("[CandidateScan] Bounded retry encountered error for %s/%s: %s", platform, content_id, retry_exc)

                        if not out_val.is_valid or not pol_val.is_valid:
                            failures = out_val.failures + pol_val.failures
                            logger.warning(
                                "[CandidateScan] Validation failed for %s/%s: %s",
                                platform, content_id, "; ".join(failures),
                            )
                            base_row["ai_recommendation"] = "AI recommendation temporarily unavailable — verified performance metrics are intact."
                            base_row["structured_analysis"] = Jsonb({
                                "validation_error": True,
                                "reasons": failures,
                                "diagnostic_status": "failed_validation",
                            })
                            base_row["llm_status"] = "failed_validation"
                        else:
                            rec = analysis.recommended_action or (
                                analysis.writer_recommendations[0]
                                if analysis.writer_recommendations
                                else "Follow up on this topic while audience interest is active."
                            )
                            base_row["ai_recommendation"] = rec
                            base_row["structured_analysis"] = Jsonb(analysis.model_dump())
                            base_row["llm_status"] = "generated"
                    else:
                        base_row["ai_recommendation"] = "AI recommendation temporarily unavailable — verified performance metrics are intact."
                        base_row["structured_analysis"] = None
                        base_row["llm_status"] = "unavailable"

                except ProviderUnavailableError:
                    base_row["ai_recommendation"] = "AI recommendation temporarily unavailable (LLM unreachable) — verified performance metrics are intact."
                    base_row["structured_analysis"] = None
                    base_row["llm_status"] = "unavailable"
                except ProviderError as p_err:
                    logger.warning("[CandidateScan] Provider error generating insight for %s/%s: %s", platform, content_id, p_err)
                    base_row["ai_recommendation"] = "AI recommendation temporarily unavailable — verified performance metrics are intact."
                    base_row["structured_analysis"] = Jsonb({
                        "provider_error": True,
                        "reasons": [str(p_err)],
                        "diagnostic_status": "failed",
                    })
                    base_row["llm_status"] = "failed"
                except Exception as exc:
                    logger.error("[CandidateScan] Unexpected error generating insight for %s/%s: %s", platform, content_id, exc)
                    base_row["ai_recommendation"] = "AI recommendation temporarily unavailable — verified performance metrics are intact."
                    base_row["structured_analysis"] = Jsonb({
                        "error": True,
                        "reasons": [str(exc)],
                        "diagnostic_status": "failed",
                    })
                    base_row["llm_status"] = "failed"

            return content_id, base_row

        tasks = [_process_candidate(c) for c in plat_candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, BaseException):
                logger.error("[CandidateScan] Error in candidate processing: %s", res)
                summary.errors += 1
                continue
            cid, row = res
            rows_to_upsert[cid] = row

            st = row.get("llm_status")
            if st == "generated":
                if cid in existing_generated and not force_refresh:
                    summary.recommendations_reused += 1
                else:
                    summary.recommendations_generated += 1
                    summary.recommendations_attempted += 1
            elif st in ("failed", "failed_validation", "unavailable"):
                summary.recommendations_failed += 1
                summary.recommendations_attempted += 1

        # Batch upsert to database
        if rows_to_upsert:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        for row in rows_to_upsert.values():
                            await cur.execute(_UPSERT_SQL, row)
                    await conn.commit()
                summary.suggestions_updated = len(rows_to_upsert)
                logger.info(
                    "[CandidateScan] %s: upserted %d suggestions (period=%s, generated=%d, reused=%d, failed=%d)",
                    platform, len(rows_to_upsert), period,
                    summary.recommendations_generated, summary.recommendations_reused, summary.recommendations_failed,
                )
            except Exception as exc:
                logger.error("[CandidateScan] Database upsert failed for %s: %s", platform, exc)
                summary.errors += len(rows_to_upsert)

        summaries.append(summary)

    return summaries

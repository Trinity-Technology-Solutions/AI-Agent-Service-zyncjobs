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
    report_period, is_low_performing, analyzed_at
) VALUES (
    %(platform)s, %(content_id)s, %(account_key)s,
    %(title)s, %(content_type)s, %(canonical_url)s, %(metric_name)s,
    %(classification)s, %(current_metric)s, %(baseline_metric)s,
    %(likes)s, %(comments)s, %(peer_explanation)s,
    %(evidence_reason)s, %(ai_recommendation)s,
    %(structured_analysis)s, %(llm_status)s,
    %(report_period)s, %(is_low_performing)s, NOW()
)
ON CONFLICT (platform, content_id, report_period) DO UPDATE SET
    account_key = EXCLUDED.account_key,
    title = EXCLUDED.title,
    content_type = EXCLUDED.content_type,
    canonical_url = EXCLUDED.canonical_url,
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
        "fbcdn.net", "cdninstagram.com", "akamaihd.net", "fbsbx.com", "cdn.", ".fbcdn."
    ]) or any(raw.lower().endswith(ext) for ext in [".mp4", ".jpg", ".jpeg", ".png", ".webp"])

    if not raw or is_cdn:
        p = platform.lower()
        ct = content_type.lower()
        if p == "youtube":
            return f"https://www.youtube.com/shorts/{content_id}" if "short" in ct else f"https://www.youtube.com/watch?v={content_id}"
        elif p == "instagram":
            return f"https://www.instagram.com/reel/{content_id}/" if "reel" in ct else f"https://www.instagram.com/p/{content_id}/"
        elif p == "facebook":
            return f"https://www.facebook.com/{content_id}"
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
                base_row["ai_recommendation"] = "AI insight generation unavailable (LLM unconfigured or unreachable)"
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
                        published_at=str(cand.get("published_at")) if cand.get("published_at") else None,
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
                        if not out_val.is_valid or not pol_val.is_valid:
                            failures = out_val.failures + pol_val.failures
                            logger.warning(
                                "[CandidateScan] Validation failed for %s/%s: %s",
                                platform, content_id, "; ".join(failures),
                            )
                            base_row["ai_recommendation"] = "AI insight validation failed"
                            base_row["structured_analysis"] = None
                            base_row["llm_status"] = "failed_validation"
                        else:
                            rec = analysis.recommended_action or (
                                analysis.writer_recommendations[0]
                                if analysis.writer_recommendations
                                else "No specific recommendation generated"
                            )
                            base_row["ai_recommendation"] = rec
                            base_row["structured_analysis"] = Jsonb(analysis.model_dump())
                            base_row["llm_status"] = "generated"
                    else:
                        base_row["ai_recommendation"] = "AI insight generation unavailable"
                        base_row["structured_analysis"] = None
                        base_row["llm_status"] = "unavailable"

                except ProviderUnavailableError:
                    base_row["ai_recommendation"] = "AI insight generation unavailable (LLM unreachable)"
                    base_row["structured_analysis"] = None
                    base_row["llm_status"] = "unavailable"
                except ProviderError as p_err:
                    logger.warning("[CandidateScan] Provider error generating insight for %s/%s: %s", platform, content_id, p_err)
                    base_row["ai_recommendation"] = "AI insight generation failed"
                    base_row["structured_analysis"] = None
                    base_row["llm_status"] = "failed"
                except Exception as exc:
                    logger.error("[CandidateScan] Unexpected error generating insight for %s/%s: %s", platform, content_id, exc)
                    base_row["ai_recommendation"] = "AI insight generation failed"
                    base_row["structured_analysis"] = None
                    base_row["llm_status"] = "failed"

            return content_id, base_row

        tasks = [_process_candidate(c) for c in plat_candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
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

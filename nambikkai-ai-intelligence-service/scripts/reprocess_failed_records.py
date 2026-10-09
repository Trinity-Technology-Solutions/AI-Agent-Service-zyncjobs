"""
Safe, idempotent, targeted reprocessing script for records that failed validation.
Supports --dry-run (default) and --apply modes.
Only touches records WHERE llm_status = 'failed_validation'.
Preserves verified performance metrics, classifications, and publication timestamps.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from psycopg.types.json import Jsonb

from app.core.config import get_settings
from app.data_sources.postgres import _get_pool, open_pool, close_pool
from app.domain.models import PerformanceCandidate
from app.providers import get_provider_with_fallback
from app.services.evidence_builder import build_evidence_package
from app.validation.output_validator import validate_output
from app.validation.policy_validator import validate_policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reprocess_failed_records")


async def reprocess(dry_run: bool = True, limit: int = 100) -> None:
    await open_pool()
    try:
        pool = _get_pool()

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, platform, content_id, account_key, title, content_type,
                           canonical_url, metric_name, classification, current_metric,
                           baseline_metric, likes, comments, peer_explanation, evidence_reason,
                           report_period, published_at, analyzed_at, llm_status
                    FROM ai_suggestions
                    WHERE llm_status = 'failed_validation'
                    ORDER BY analyzed_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = await cur.fetchall()
                cols = [d[0] for d in cur.description] if cur.description else []
                failed_records = [dict(zip(cols, r)) for r in rows]

        total = len(failed_records)
        print(f"\n{'='*70}")
        print(f"TARGETED REPROCESSING PLAN {'(DRY RUN)' if dry_run else '(APPLY MODE)'}")
        print(f"{'='*70}")
        print(f"Found {total} records with llm_status = 'failed_validation'.")

        if total == 0:
            print("No failed records found to reprocess.")
            return

        # Breakdown by platform and report_period
        breakdown: dict[str, int] = {}
        for r in failed_records:
            k = f"{r['platform']} | {r['report_period']} | {r['classification']}"
            breakdown[k] = breakdown.get(k, 0) + 1

        print("\nBreakdown of targets:")
        for k, cnt in sorted(breakdown.items()):
            print(f"  {k}: {cnt} items")

        print("\nSample records to reprocess:")
        for r in failed_records[:5]:
            pub_str = r['published_at'].strftime('%Y-%m-%d') if r.get('published_at') else 'N/A'
            print(f"  ID={r['id']} {r['platform']}/{r['content_id']} ({r['report_period']}) pub={pub_str} metric={r['current_metric']} likes={r['likes']}")

        if dry_run:
            print(f"\n[DRY RUN] Complete. No changes made to database.")
            print(f"Run with --apply to re-run generation and validation through the repaired pipeline.")
            return

        print(f"\nStarting re-analysis of {total} failed records using repaired validation pipeline...")
        provider, fallback_provider = get_provider_with_fallback()
        if provider is None:
            print("ERROR: No LLM provider configured or available. Aborting.")
            return

        success_count = 0
        fail_count = 0

        for idx, r in enumerate(failed_records, start=1):
            cid = r["content_id"]
            plat = r["platform"]
            print(f"[{idx}/{total}] Processing {plat}/{cid} ({r['classification']})...", end="", flush=True)

            cand_obj = PerformanceCandidate(
                platform=plat,
                content_id=cid,
                account_key=r.get("account_key") or "unknown",
                performance_level=r["classification"],
                content_type=r.get("content_type") or "Video",
                title=r.get("title") or cid,
                url=r.get("canonical_url"),
                canonical_url=r.get("canonical_url"),
                published_at=r["published_at"].isoformat() if r.get("published_at") else None,
                period=r["report_period"],
                current_metric=float(r["current_metric"] or 0),
                baseline_metric=float(r["baseline_metric"]) if r.get("baseline_metric") is not None else None,
                metric_name=r.get("metric_name") or "views",
                likes=float(r["likes"] or 0),
                comments=float(r["comments"] or 0),
                peer_explanation=r.get("peer_explanation") or "",
            )
            evidence = build_evidence_package(candidate=cand_obj)

            analysis = None
            try:
                analysis = await provider.generate_structured_analysis(evidence)
            except Exception as e:
                if fallback_provider:
                    try:
                        analysis = await fallback_provider.generate_structured_analysis(evidence)
                    except Exception as fb_e:
                        logger.warning("Fallback provider failed: %s", fb_e)

            if analysis:
                ov = validate_output(analysis, evidence)
                pv = validate_policy(analysis)

                # Bounded retry if initial validation failed
                if not ov.is_valid or not pv.is_valid:
                    first_failures = ov.failures + pv.failures
                    feedback = "; ".join(first_failures)
                    try:
                        retry_analysis = await provider.generate_structured_analysis(evidence, feedback=feedback)
                        retry_ov = validate_output(retry_analysis, evidence)
                        retry_pv = validate_policy(retry_analysis)
                        if retry_ov.is_valid and retry_pv.is_valid:
                            analysis = retry_analysis
                            ov = retry_ov
                            pv = retry_pv
                    except Exception as r_exc:
                        logger.warning("Retry error: %s", r_exc)

                if ov.is_valid and pv.is_valid:
                    rec = analysis.recommended_action or analysis.writer_recommendations[0]
                    sa_dump = Jsonb(analysis.model_dump())
                    async with pool.connection() as conn:
                        async with conn.cursor() as cur:
                            await cur.execute(
                                """
                                UPDATE ai_suggestions
                                SET ai_recommendation = %s,
                                    structured_analysis = %s,
                                    llm_status = 'generated'
                                WHERE id = %s
                                """,
                                (rec, sa_dump, r["id"]),
                            )
                            await conn.commit()
                    print(" OK (generated)")
                    success_count += 1
                else:
                    failures = ov.failures + pv.failures
                    print(f" FAILED VALIDATION: {failures}")
                    fail_count += 1
            else:
                print(" PROVIDER ERROR")
                fail_count += 1

        print(f"\nReprocessing complete: {success_count} succeeded, {fail_count} failed.")
    finally:
        await close_pool()


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser(description="Reprocess failed validation records")
    parser.add_argument("--apply", action="store_true", help="Apply updates to database (default is dry-run)")
    parser.add_argument("--limit", type=int, default=100, help="Maximum records to process")
    args = parser.parse_args()

    asyncio.run(reprocess(dry_run=not args.apply, limit=args.limit))


if __name__ == "__main__":
    main()

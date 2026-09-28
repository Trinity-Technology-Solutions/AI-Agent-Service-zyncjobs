"""
Evidence builder for AI Performance Insights.

Builds a bounded evidence package strictly from verified dashboard candidate facts:
- platform, content_id, account_key
- performance_level (HIGH_PERFORMING | LOW_PERFORMING)
- content_type (Video/Short, Post/Reel)
- canonical_url
- verified metrics (views/reach, likes, comments)
- peer comparison explanation
- transcript (full available transcript, up to TRANSCRIPT_MAX_CHARS configured chars)
- selected analysis period (7d, 30d, 90d)

The evidence builder does NOT calculate, classify, gate, or reinterpret performance.

Transcript handling:
- The complete raw transcript is preserved in the PerformanceCandidate object.
- The transcript forwarded to the LLM is bounded by TRANSCRIPT_MAX_CHARS (default 8000).
- This is NOT an arbitrary 2500-char truncation: TRANSCRIPT_MAX_CHARS is configurable
  and sized to fit within the configured model's context window.
- If a transcript exceeds the limit, the excerpt starts from the beginning so the
  hook/opening (most editorially important section) is always included.
- The truncation is clearly documented in the evidence package so the LLM knows
  it received an excerpt rather than the full transcript.
- Set TRANSCRIPT_MAX_CHARS=0 in .env to disable the limit for large-context models.
"""
from typing import Optional, Any
from app.core.config import get_settings
from app.domain.models import (
    DataQuality,
    EvidencePackage,
    PerformanceCandidate,
)


def _get_transcript_limit() -> int:
    """Return the configured transcript character limit. 0 means no limit."""
    try:
        return get_settings().TRANSCRIPT_MAX_CHARS
    except Exception:
        return 8000  # safe fallback


def _bound_transcript(text: str) -> tuple[str, bool]:
    """
    Apply the configured transcript character limit.

    Returns:
        (bounded_text, was_truncated)

    The full transcript is always preserved in the candidate; this function only
    determines what to forward to the LLM prompt.
    """
    limit = _get_transcript_limit()
    if limit <= 0 or len(text) <= limit:
        return text, False
    return text[:limit], True


def build_evidence_package(
    candidate: PerformanceCandidate | dict[str, Any],
    transcript_excerpt: Optional[str] = None,
    regional_signals: Optional[dict] = None,
    **kwargs: Any,
) -> EvidencePackage:
    """
    Construct a clean, bounded EvidencePackage from an authoritative performance candidate.

    The transcript passed to the LLM is bounded by TRANSCRIPT_MAX_CHARS (env-configurable,
    default 8000 characters) rather than the previous hardcoded 2500-character limit.
    This allows the LLM to access much more of the available transcript content.

    If the transcript exceeds the configured limit, the excerpt starts from the beginning
    so the hook/opening (most editorially significant) is always included.
    """
    if isinstance(candidate, dict):
        # Support dict input for convenience
        cand_obj = PerformanceCandidate(
            platform=str(candidate.get("platform") or "").lower().strip(),
            content_id=str(candidate.get("content_id") or "").strip(),
            account_key=str(candidate.get("account_key") or "").strip(),
            performance_level=str(candidate.get("performance_level") or "HIGH_PERFORMING").upper(),
            content_type=str(candidate.get("content_type") or "Video"),
            title=str(candidate.get("title") or "")[:200],
            url=candidate.get("url") or candidate.get("canonical_url"),
            canonical_url=candidate.get("canonical_url") or candidate.get("url"),
            published_at=str(candidate.get("published_at")) if candidate.get("published_at") else None,
            period=str(candidate.get("period") or "30d"),
            current_metric=float(candidate.get("current_metric") or 0.0),
            baseline_metric=float(candidate["baseline_metric"]) if candidate.get("baseline_metric") is not None else None,
            metric_name=str(candidate.get("metric_name") or "views"),
            likes=float(candidate.get("likes") or 0.0),
            comments=float(candidate.get("comments") or 0.0),
            peer_explanation=str(candidate.get("peer_explanation") or ""),
            transcript=candidate.get("transcript"),
            transcript_status=candidate.get("transcript_status"),
        )
    else:
        cand_obj = candidate

    # Determine transcript to include in the LLM prompt.
    # Priority: explicitly supplied transcript_excerpt > candidate.transcript.
    # The full transcript remains on cand_obj.transcript regardless.
    raw_transcript = transcript_excerpt or cand_obj.transcript

    if raw_transcript:
        llm_transcript, was_truncated = _bound_transcript(raw_transcript)
        limit = _get_transcript_limit()
        if was_truncated:
            # Append a clear disclosure so the LLM knows this is an excerpt.
            note = (
                f"\n\n[TRANSCRIPT NOTE: The full transcript is {len(raw_transcript):,} characters. "
                f"This excerpt shows the first {limit:,} characters. "
                f"Remaining content was omitted to fit the model context window. "
                f"Analyze based on the available excerpt and note this limitation.]"
            )
            llm_transcript = llm_transcript + note
    else:
        llm_transcript = None

    notes = cand_obj.peer_explanation if cand_obj.peer_explanation else None

    quality = DataQuality(
        baseline_available=True,
        metrics_complete=cand_obj.current_metric >= 0,
        notes=notes,
    )

    return EvidencePackage(
        candidate=cand_obj,
        transcript_excerpt=llm_transcript,
        regional_signals=regional_signals,
        data_quality=quality,
    )

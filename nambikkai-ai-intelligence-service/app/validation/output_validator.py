from __future__ import annotations

import re
from app.domain.models import EditorialAnalysis, EvidencePackage, ValidationResult


def validate_output(
    analysis: EditorialAnalysis,
    evidence: EvidencePackage,
) -> ValidationResult:
    failures: list[str] = []

    # 1. Essential schema validation
    if not (0.0 <= analysis.confidence <= 1.0):
        failures.append(f"confidence {analysis.confidence} out of range [0, 1]")

    if not analysis.content_intent or not analysis.content_intent.strip():
        failures.append("content_intent is empty")

    if not analysis.observed_signals or not any(s.strip() for s in analysis.observed_signals):
        failures.append("observed_signals is empty")

    if not analysis.writer_recommendations or not any(r.strip() for r in analysis.writer_recommendations):
        failures.append("writer_recommendations is empty")

    cand = getattr(evidence, "candidate", None)
    rec_act = (analysis.recommended_action or "").lower()
    factors_text = " ".join(analysis.possible_contributing_factors or []).lower()
    all_signals_text = " ".join(analysis.observed_signals or []).lower()
    all_recs_text = f"{' '.join(analysis.writer_recommendations or []).lower()} {rec_act}"
    all_analysis_text = f"{all_signals_text} {all_recs_text} {factors_text}"

    if cand:
        # 2. Performance classification alignment
        perf_level = (cand.performance_level or "").upper()
        if perf_level == "HIGH_PERFORMING":
            severe_negatives = [
                "disastrous flop", "terrible failure", "failed completely",
                "severe underperformance", "complete flop"
            ]
            for term in severe_negatives:
                if term in all_analysis_text:
                    failures.append(f"Classification mismatch: HIGH_PERFORMING content described with '{term}'")
        elif perf_level == "LOW_PERFORMING":
            wild_positives = [
                "massive viral success", "broke all records",
                "wildly explosive reach", "viral runaway hit"
            ]
            for term in wild_positives:
                if term in all_analysis_text:
                    failures.append(f"Classification mismatch: LOW_PERFORMING content described with '{term}'")

        # 3. Transcript grounding
        has_transcript = bool(evidence.transcript_excerpt and evidence.transcript_excerpt.strip()) or bool(cand.transcript and cand.transcript.strip())
        if not has_transcript:
            transcript_claims = [
                "the speaker in the transcript says",
                "in the transcript, they say",
                "according to the transcript dialogue",
                "the narrator explicitly said",
            ]
            for claim in transcript_claims:
                if claim in all_analysis_text:
                    failures.append("Fabricated transcript: claims transcript dialogue when no transcript was provided")

        # 4. Content / Platform identity
        platform = (cand.platform or "").lower()
        if platform == "instagram":
            if "youtube thumbnail" in all_analysis_text or "youtube channel" in all_analysis_text:
                failures.append("Platform mismatch: YouTube-specific advice given for Instagram content")
        elif platform == "youtube":
            if "instagram reel" in all_analysis_text or "instagram grid" in all_analysis_text or "link in bio" in all_analysis_text:
                failures.append("Platform mismatch: Instagram-specific advice given for YouTube content")

        # 5. Numerical sanity (Gross hallucination safeguard)
        curr_metric = float(cand.current_metric or 0)
        if curr_metric < 500000:
            large_number_patterns = [
                r'\b\d+\s*million\b', r'\b\d+m\s+views\b',
                r'\b\d+m\s+reach\b', r'\b[1-9]\d{6,}\b'
            ]
            for pat in large_number_patterns:
                if re.search(pat, all_analysis_text):
                    failures.append(f"Fabricated metric: claims millions for content with verified metric {curr_metric:,.0f}")
                    break

    return ValidationResult(is_valid=len(failures) == 0, failures=failures)


"""Hallucination validator — basic factual consistency check against source context."""
import logging
import re
from typing import Any
from recruitment_ai.brains.shared.brain_state import BrainState, RetrievedDocuments
from recruitment_ai.brains.shared.brain_result import BrainResult

logger = logging.getLogger(__name__)


def extract_noun_phrases(text: str) -> set[str]:
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    return {w.lower() for w in words if len(w) > 2}


def extract_fact_claims(text: str) -> list[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def check_claims_against_context(claims: list[str], context_texts: list[str]) -> list[dict]:
    issues = []
    context_combined = " ".join(context_texts).lower()
    context_entities = extract_noun_phrases(context_combined)

    for claim in claims:
        claim_lower = claim.lower()
        claim_entities = extract_noun_phrases(claim)

        if not claim_entities:
            continue

        unsupported = [e for e in claim_entities if e not in context_entities and len(e) > 3]

        numeric_claims = re.findall(r"\b(\d{3,})\b", claim)
        unsupported_numeric = []
        for num in numeric_claims:
            if num not in context_combined:
                unsupported_numeric.append(num)

        if unsupported or unsupported_numeric:
            issues.append({
                "claim": claim[:100],
                "unsupported_entities": unsupported[:5],
                "unsupported_numbers": unsupported_numeric[:3],
                "severity": "low" if len(unsupported) <= 2 else "medium",
            })

    return issues


def validate_against_context(result: BrainResult, state: BrainState) -> list[dict]:
    response_text = ""
    if isinstance(result.response, dict):
        response_text = str(result.response.get("reply", "") or result.response.get("text", "") or result.response.get("response", "") or "")
    elif isinstance(result.response, str):
        response_text = result.response

    if not response_text:
        return []

    rag_chunks = state.retrieved_documents.chunks or state.context.get("rag_context", [])
    context_texts = [c.get("text", "") or c.get("content", "") or str(c) for c in rag_chunks]

    if not context_texts:
        return []

    claims = extract_fact_claims(response_text)
    return check_claims_against_context(claims, context_texts)


def has_hallucination_risk(result: BrainResult, state: BrainState, threshold: int = 3) -> bool:
    issues = validate_against_context(result, state)
    high_severity = [i for i in issues if i.get("severity") in ("medium", "high")]
    return len(high_severity) >= threshold

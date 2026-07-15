"""Citation validator — validates source attribution format and consistency."""
import logging
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_CITATION_FIELDS = {"title", "url"}
OPTIONAL_CITATION_FIELDS = {"snippet", "source", "relevance_score", "category"}


def validate_citation(citation: dict) -> list[str]:
    errors = []
    if not isinstance(citation, dict):
        return ["Citation must be a dict"]
    for field in REQUIRED_CITATION_FIELDS:
        if field not in citation:
            errors.append(f"Missing required citation field: {field}")
    if "url" in citation and citation["url"]:
        url = str(citation["url"])
        if not url.startswith(("http://", "https://", "/")):
            errors.append(f"Invalid citation URL: {url}")
    if "relevance_score" in citation:
        score = citation["relevance_score"]
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            errors.append(f"relevance_score must be 0-1, got {score}")
    if "title" in citation and not citation["title"]:
        errors.append("Citation title cannot be empty")
    return errors


def validate_citations(citations: list[dict]) -> list[str]:
    all_errors = []
    for i, c in enumerate(citations):
        errors = validate_citation(c)
        for e in errors:
            all_errors.append(f"citation[{i}]: {e}")
    return all_errors


def check_response_citation_consistency(response_text: str, citations: list[dict]) -> list[str]:
    warnings = []
    cited_titles = {c.get("title", "").strip().lower() for c in citations if c.get("title")}
    if not cited_titles:
        return warnings
    for title in cited_titles:
        if title and title not in response_text.lower():
            warnings.append(f"Citation '{title}' not referenced in response")
    return warnings


def deduplicate_citations(citations: list[dict]) -> list[dict]:
    seen_urls = set()
    deduped = []
    for c in citations:
        url = c.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            deduped.append(c)
        elif not url:
            deduped.append(c)
    return deduped

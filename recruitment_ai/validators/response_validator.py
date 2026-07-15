"""Response validator — validates BrainResult responses for required fields, structure, and completeness."""
import logging
from typing import Any
from recruitment_ai.brains.shared.brain_result import BrainResult
from recruitment_ai.brains.shared.brain_state import BrainState
from recruitment_ai.validators.json_validator import validate_json
from recruitment_ai.validators.citation_validator import validate_citations
from recruitment_ai.validators.safety_validator import validate_safety
from recruitment_ai.validators.schema_validator import validate_brain_result, validate_brain_state
from recruitment_ai.validators.hallucination_validator import validate_against_context

logger = logging.getLogger(__name__)

INTENT_REQUIRED_FIELDS = {
    "RESUME_PARSER": ["personal_info", "skills", "experience"],
    "ATS_SCORE": ["ats_score", "keyword_match", "suggestions"],
    "JOB_MATCH": ["match_score", "skill_match", "recommendation"],
    "JOB_PARSER": ["title", "skills", "description"],
    "JD_GENERATOR": ["title", "responsibilities", "requirements"],
    "CAREER_ADVICE": ["career_path", "skill_gaps", "advice"],
    "SKILL_ASSESSMENT": ["questions"],
    "INTERVIEW_PREP": ["questions", "topics_to_review"],
    "CAREER_ROADMAP": ["career_path", "timeline_months"],
    "SKILL_GAP": ["skill_gaps", "recommended_courses"],
    "RESUME_BUILDER": ["summary", "ats_keywords"],
    "COVER_LETTER": ["content"],
    "RECRUITER": ["candidates", "summary"],
    "RECRUITER_SHORTLIST": ["candidates", "rankings"],
}

REQUIRED_REPLY_FIELDS = {"reply", "sources"}


class ValidationReport:
    def __init__(self):
        self.passed: bool = True
        self.brain_errors: list[str] = []
        self.schema_errors: list[str] = []
        self.citation_errors: list[str] = []
        self.safety_warnings: list[str] = []
        self.hallucination_warnings: list[str] = []
        self.field_warnings: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.passed = False
        self.brain_errors.append(msg)

    def add_schema_error(self, msg: str) -> None:
        self.passed = False
        self.schema_errors.append(msg)

    def add_citation_error(self, msg: str) -> None:
        self.citation_errors.append(msg)

    def add_safety_warning(self, msg: str) -> None:
        self.safety_warnings.append(msg)

    def add_hallucination_warning(self, msg: str) -> None:
        self.hallucination_warnings.append(msg)

    def add_field_warning(self, msg: str) -> None:
        self.field_warnings.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def all_issues(self) -> list[str]:
        return (
            self.brain_errors
            + self.schema_errors
            + self.citation_errors
            + self.safety_warnings
            + self.hallucination_warnings
            + self.field_warnings
            + self.warnings
        )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "brain_errors": self.brain_errors,
            "schema_errors": self.schema_errors,
            "citation_errors": self.citation_errors,
            "safety_warnings": self.safety_warnings,
            "hallucination_warnings": self.hallucination_warnings,
            "field_warnings": self.field_warnings,
        }


def validate_response(result: BrainResult, state: BrainState) -> ValidationReport:
    report = ValidationReport()

    brain_errors = validate_brain_result(result)
    for e in brain_errors:
        report.add_schema_error(e)

    state_errors = validate_brain_state(state)
    for e in state_errors:
        report.add_schema_error(e)

    citation_errors = validate_citations(result.citations)
    for e in citation_errors:
        report.add_citation_error(e)

    response = result.response or {}
    intent = state.intent or "CHAT"

    if intent == "CHAT" and response:
        missing = REQUIRED_REPLY_FIELDS - set(response.keys())
        for f in missing:
            report.add_field_warning(f"CHAT response missing expected field: {f}")

    if intent in INTENT_REQUIRED_FIELDS and response:
        expected = set(INTENT_REQUIRED_FIELDS[intent])
        missing = expected - set(response.keys())
        for f in missing:
            report.add_field_warning(f"{intent} response missing expected field: {f}")

    if result.response:
        text = str(result.response)
        safety = validate_safety(text)
        for issue in safety.issues:
            report.add_safety_warning(f"{issue['category']}: {issue['detail']}")

    returns_citations = bool(result.citations)
    has_rag = bool(state.retrieved_documents.chunks)
    if has_rag and not returns_citations:
        report.add_warning("RAG context loaded but no citations returned")

    hallucination_issues = validate_against_context(result, state)
    for h in hallucination_issues:
        report.add_hallucination_warning(
            f"Unsupported entities: {h['unsupported_entities']} in claim: {h['claim'][:60]}..."
        )

    return report

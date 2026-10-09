from app.domain.models import EditorialAnalysis, ValidationResult

# Absolute, unproven causal certainty is forbidden.
# Natural conversational language (e.g. "likely because of", "caused by the strong visual hook",
# "due to", "resulting in") is fully permitted for clear editorial communication.
_ABSOLUTE_CAUSAL_PHRASES = [
    "proven beyond doubt",
    "definitely caused by",
    "scientific proof that",
    "100% caused by",
    "guaranteed reason",
    "indisputable cause",
    "undeniable proof that",
]


def validate_policy(analysis: EditorialAnalysis) -> ValidationResult:
    failures: list[str] = []
    all_text = " ".join(
        analysis.possible_contributing_factors + analysis.observed_signals
    ).lower()
    for phrase in _ABSOLUTE_CAUSAL_PHRASES:
        if phrase in all_text:
            failures.append(f"Policy violation: absolute causal certainty detected — '{phrase}'")
    return ValidationResult(is_valid=len(failures) == 0, failures=failures)


from app.domain.models import EditorialAnalysis, ValidationResult

_FORBIDDEN_CAUSAL = ["caused by", "because of", "directly resulted", "proven to"]


def validate_policy(analysis: EditorialAnalysis) -> ValidationResult:
    failures: list[str] = []
    all_text = " ".join(
        analysis.possible_contributing_factors + analysis.observed_signals
    ).lower()
    for phrase in _FORBIDDEN_CAUSAL:
        if phrase in all_text:
            failures.append(f"Policy violation: causal language detected — '{phrase}'")
    return ValidationResult(is_valid=len(failures) == 0, failures=failures)

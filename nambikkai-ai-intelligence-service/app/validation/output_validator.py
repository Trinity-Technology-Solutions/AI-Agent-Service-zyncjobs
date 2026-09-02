from app.domain.models import EditorialAnalysis, EvidencePackage, ValidationResult


def validate_output(
    analysis: EditorialAnalysis,
    evidence: EvidencePackage,
) -> ValidationResult:
    failures: list[str] = []

    if not (0.0 <= analysis.confidence <= 1.0):
        failures.append(f"confidence {analysis.confidence} out of range [0, 1]")

    if not analysis.content_intent or not analysis.content_intent.strip():
        failures.append("content_intent is empty")

    if not analysis.observed_signals:
        failures.append("observed_signals is empty")

    if not analysis.writer_recommendations:
        failures.append("writer_recommendations is empty")

    return ValidationResult(is_valid=len(failures) == 0, failures=failures)

from app.domain.models import GateClassification, GateResult


def should_invoke_llm(gate_result: GateResult) -> bool:
    """Only invoke the LLM for BOOMING_SURGE events."""
    return gate_result.classification == GateClassification.BOOMING_SURGE


def should_monitor(gate_result: GateResult) -> bool:
    return gate_result.classification == GateClassification.ELEVATED

from app.domain.models import GateClassification, GateResult

_LLM_ELIGIBLE = {GateClassification.BOOMING_SURGE, GateClassification.SURGE_CANDIDATE}


def should_invoke_llm(gate_result: GateResult) -> bool:
    """Invoke the LLM for BOOMING_SURGE (confirmed surge) and SURGE_CANDIDATE (early signal)."""
    return gate_result.llm_eligible


def should_monitor(gate_result: GateResult) -> bool:
    return gate_result.classification == GateClassification.ELEVATED

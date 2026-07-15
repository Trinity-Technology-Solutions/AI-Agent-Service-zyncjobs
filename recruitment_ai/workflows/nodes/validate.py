"""Validate node — runs full validation pipeline on every response.
Ensures consistent shape, safety, citations, and hallucination checks.
"""
import logging
from recruitment_ai.brains.base import BrainState, BrainResult
from recruitment_ai.validators.response_validator import validate_response, ValidationReport

logger = logging.getLogger(__name__)


async def validate_node(state: BrainState) -> BrainState:
    if state.error and not state.result:
        state.result = {"error": state.error}
        return state

    from recruitment_ai.brains.shared.brain_result import BrainResult
    result = BrainResult(
        success=not bool(state.error),
        response=state.result,
        metadata=state.metadata,
    )

    report = validate_response(result, state)

    if not report.passed:
        issues = report.all_issues
        logger.warning("Validation failed: %s", issues[:3])
        state.metadata["validation"] = report.to_dict()
        state.metadata["validation_warnings"] = issues
    else:
        state.metadata["validation"] = {"passed": True}

    return state

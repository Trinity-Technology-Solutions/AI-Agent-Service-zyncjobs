"""Schema validator — validates BrainResult / BrainState against expected Pydantic schemas."""
import logging
from typing import Any
from pydantic import BaseModel, ValidationError
from recruitment_ai.brains.shared.brain_result import BrainResult
from recruitment_ai.brains.shared.brain_state import BrainState

logger = logging.getLogger(__name__)


def validate_brain_result(result: BrainResult) -> list[str]:
    errors = []
    if not result.success and not result.response:
        errors.append("Brain failed with no response payload")
    if result.response is not None and not isinstance(result.response, dict):
        errors.append(f"Response must be a dict, got {type(result.response).__name__}")
    if result.execution_time < 0:
        errors.append(f"Negative execution_time: {result.execution_time}")
    if result.tokens < 0:
        errors.append(f"Negative token count: {result.tokens}")
    for w in result.warnings:
        if not isinstance(w, str):
            errors.append(f"Warning must be string, got {type(w).__name__}")
    return errors


def validate_brain_state(state: BrainState) -> list[str]:
    errors = []
    if state.request and state.request.query and len(state.request.query) > 10000:
        errors.append(f"Query exceeds 10k chars ({len(state.request.query)})")
    if state.error and not state.response:
        pass
    if state.response and not isinstance(state.response, dict):
        errors.append(f"Response must be a dict, got {type(state.response).__name__}")
    return errors


def validate_against_schema(data: dict, schema: type[BaseModel]) -> list[str]:
    try:
        schema(**data)
        return []
    except ValidationError as e:
        return [f"{err['loc']}: {err['msg']}" for err in e.errors()]


def coerce_to_schema(data: dict, schema: type[BaseModel]) -> BaseModel | None:
    try:
        return schema(**data)
    except ValidationError as e:
        logger.warning("Schema coercion failed: %s", e)
        return None

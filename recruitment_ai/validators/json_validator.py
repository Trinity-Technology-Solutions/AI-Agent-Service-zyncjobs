"""JSON validator — parses and validates LLM JSON output with graceful fallback."""
import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"```(?:json)?\s*\n?(.*?)```",
        r"\{[^{}]*\}",
    ]
    for p in patterns:
        match = re.search(p, text, re.DOTALL)
        if match:
            candidate = match.group(1) if match.lastindex else match.group(0)
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None


def validate_json(text: str) -> dict | list | None:
    extracted = extract_json(text)
    if not extracted:
        logger.warning("No valid JSON found in LLM output")
        return None
    try:
        return json.loads(extracted)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed: %s", e)
        return None


def validate_json_strict(text: str, schema_type: str = "object") -> dict | list | None:
    result = validate_json(text)
    if result is None:
        return None
    if schema_type == "object" and not isinstance(result, dict):
        logger.warning("Expected JSON object, got %s", type(result).__name__)
        return None
    if schema_type == "array" and not isinstance(result, list):
        logger.warning("Expected JSON array, got %s", type(result).__name__)
        return None
    return result


def ensure_json_fields(data: dict, required: list[str]) -> list[str]:
    missing = [f for f in required if f not in data]
    if missing:
        logger.warning("Missing required JSON fields: %s", missing)
    return missing


def sanitize_json(data: Any, default: Any = None) -> Any:
    try:
        json.dumps(data)
        return data
    except (TypeError, ValueError):
        return default

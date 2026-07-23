"""JSON validator — parses and validates LLM JSON output with graceful fallback."""
import json
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str | None:
    if not text:
        return None
    # First try the full text directly — most LLMs follow "No extra text" instruction
    text_stripped = text.strip()
    if text_stripped.startswith("{") and text_stripped.endswith("}"):
        try:
            json.loads(text_stripped)
            return text_stripped
        except json.JSONDecodeError:
            pass
    # Fallback: extract from markdown code blocks
    patterns = [
        r"```(?:json)?\s*\n?(.*?)```",
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
    # Last resort: find outermost balanced JSON via brace counting
    try:
        return _extract_outermost_json(text)
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def _extract_outermost_json(text: str) -> str | None:
    """Extract the outermost balanced JSON object/array using brace counting."""
    brace_depth = 0
    bracket_depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if start == -1:
                start = i
            brace_depth += 1
        elif ch == '}':
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                candidate = text[start:i+1]
                json.loads(candidate)
                return candidate
        elif ch == '[':
            if start == -1:
                start = i
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
            if bracket_depth == 0 and start >= 0:
                candidate = text[start:i+1]
                json.loads(candidate)
                return candidate
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

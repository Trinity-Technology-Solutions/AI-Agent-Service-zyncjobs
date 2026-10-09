from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.domain.models import EditorialAnalysis, EvidencePackage

logger = logging.getLogger(__name__)


def extract_json(raw: str) -> dict[str, Any]:
    """
    Robustly extract a JSON object from LLM response text.
    Handles raw JSON, markdown code fences (```json ... ```),
    and JSON objects surrounded by conversational introductory or concluding text.
    """
    s = raw.strip()
    # 1. Try direct json.loads
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Try regex extraction of markdown code blocks
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', s)
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 3. Find outermost curly braces
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Could not extract valid JSON object from LLM response", s, 0)


def extract_and_normalize_analysis(
    raw_content: str,
    evidence: EvidencePackage,
    request_id: str = "",
) -> EditorialAnalysis:
    """
    Extract JSON and normalize fields into EditorialAnalysis.
    Tolerates minor stylistic and formatting differences (e.g. single strings
    instead of lists, percentage confidence strings) while strictly preserving
    the verified schema.
    """
    data = extract_json(raw_content)

    cand = getattr(evidence, "candidate", None)
    title = cand.title if cand else evidence.content_metadata.title

    # 1. Confidence normalization
    conf = data.get("confidence")
    if conf is None:
        data["confidence"] = 0.8
    else:
        try:
            conf_val = float(str(conf).replace("%", "").strip())
            if 1.0 < conf_val <= 100.0:
                conf_val /= 100.0
            data["confidence"] = max(0.0, min(1.0, conf_val))
        except Exception:
            data["confidence"] = 0.8

    # 2. Content intent
    ci = data.get("content_intent")
    if not ci or not str(ci).strip():
        data["content_intent"] = f"Content discussing '{title}'."
    else:
        data["content_intent"] = str(ci).strip()

    # 3. Observed signals (must be non-empty list of strings)
    signals = data.get("observed_signals")
    if isinstance(signals, str):
        signals = [signals]
    elif not isinstance(signals, list) or not signals:
        signals = [f"Verified performance metrics evaluated for '{title}'."]
    data["observed_signals"] = [str(s).strip() for s in signals if str(s).strip()]
    if not data["observed_signals"]:
        data["observed_signals"] = [f"Verified performance metrics evaluated for '{title}'."]

    # 4. Possible contributing factors (list of strings)
    factors = data.get("possible_contributing_factors")
    if isinstance(factors, str):
        factors = [factors]
    elif not isinstance(factors, list) or not factors:
        factors = ["Performance influenced by presentation, topic resonance, and viewer interest."]
    data["possible_contributing_factors"] = [str(f).strip() for f in factors if str(f).strip()]
    if not data["possible_contributing_factors"]:
        data["possible_contributing_factors"] = ["Performance influenced by presentation, topic resonance, and viewer interest."]

    # 5. Writer recommendations (must be non-empty list of strings)
    recs = data.get("writer_recommendations")
    if isinstance(recs, str):
        recs = [recs]
    elif not isinstance(recs, list) or not recs:
        rec_act = data.get("recommended_action")
        if rec_act and str(rec_act).strip():
            recs = [str(rec_act).strip()]
        else:
            recs = ["Follow up on this topic while audience interest is active."]
    data["writer_recommendations"] = [str(r).strip() for r in recs if str(r).strip()]
    if not data["writer_recommendations"]:
        data["writer_recommendations"] = ["Follow up on this topic while audience interest is active."]

    # 6. Recommended action (single string)
    rec_action = data.get("recommended_action")
    if not rec_action or not str(rec_action).strip():
        data["recommended_action"] = data["writer_recommendations"][0]
    else:
        data["recommended_action"] = str(rec_action).strip()

    # 7. Helper array fields
    for list_field in (
        "keyword_suggestions", "title_suggestions", "description_suggestions",
        "hashtag_suggestions", "cross_platform_ideas", "limitations"
    ):
        val = data.get(list_field)
        if isinstance(val, str):
            data[list_field] = [val.strip()]
        elif not isinstance(val, list):
            data[list_field] = []

    return EditorialAnalysis(**data)

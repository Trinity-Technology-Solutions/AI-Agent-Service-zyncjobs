"""Audit logger — immutable audit trail for compliance and debugging."""
import logging
import json
from datetime import datetime, timezone
from typing import Optional
from recruitment_ai.brains.shared.brain_state import BrainState
from recruitment_ai.brains.shared.brain_result import BrainResult

logger = logging.getLogger("audit")


class AuditLogger:
    def log_request(self, state: BrainState) -> None:
        logger.info(json.dumps({
            "event": "audit_request",
            "session_id": state.session.id,
            "user_id": state.user.id,
            "user_role": state.user.role,
            "intent": state.intent,
            "query_preview": (state.request.query or state.query or "")[:200],
            "has_file": bool(state.request.file_content or state.file_content),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, default=str))

    def log_response(self, state: BrainState, result: BrainResult) -> None:
        response_preview = ""
        if isinstance(result.response, dict):
            response_preview = json.dumps(result.response)[:200]
        elif isinstance(result.response, str):
            response_preview = result.response[:200]

        logger.info(json.dumps({
            "event": "audit_response",
            "session_id": state.session.id,
            "user_id": state.user.id,
            "intent": state.intent,
            "brain": state.execution.brain or result.metadata.get("brain", ""),
            "success": result.success,
            "response_preview": response_preview,
            "citation_count": len(result.citations),
            "tokens": result.tokens,
            "duration_ms": state.execution.duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, default=str))

    def log_error(self, state: BrainState, error: str) -> None:
        logger.error(json.dumps({
            "event": "audit_error",
            "session_id": state.session.id,
            "user_id": state.user.id,
            "intent": state.intent,
            "brain": state.execution.brain or "",
            "error": error[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, default=str))


audit_logger = AuditLogger()

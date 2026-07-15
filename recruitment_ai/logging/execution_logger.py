"""Execution logger — logs brain execution details (start, end, success, error)."""
import logging
import json
import time
from datetime import datetime, timezone
from typing import Optional
from recruitment_ai.brains.shared.brain_state import BrainState
from recruitment_ai.brains.shared.brain_result import BrainResult

logger = logging.getLogger("execution")


class ExecutionLogger:
    def log_start(self, state: BrainState) -> str:
        execution_id = f"{state.session.id or 'anon'}_{int(time.time() * 1000)}"
        logger.info(json.dumps({
            "event": "execution_start",
            "execution_id": execution_id,
            "session_id": state.session.id,
            "user_id": state.user.id,
            "intent": state.intent,
            "query_preview": (state.request.query or state.query or "")[:100],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        return execution_id

    def log_end(self, state: BrainState, result: BrainResult, execution_id: str) -> None:
        logger.info(json.dumps({
            "event": "execution_end",
            "execution_id": execution_id,
            "session_id": state.session.id,
            "user_id": state.user.id,
            "intent": state.intent,
            "brain": state.execution.brain or result.metadata.get("brain", ""),
            "success": result.success,
            "duration_ms": state.execution.duration_ms,
            "tokens": result.tokens,
            "cost": result.cost,
            "cache_hit": state.execution.cache_hit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    def log_error(self, state: BrainState, error: str, execution_id: str) -> None:
        logger.error(json.dumps({
            "event": "execution_error",
            "execution_id": execution_id,
            "session_id": state.session.id,
            "user_id": state.user.id,
            "intent": state.intent,
            "error": error[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))


execution_logger = ExecutionLogger()

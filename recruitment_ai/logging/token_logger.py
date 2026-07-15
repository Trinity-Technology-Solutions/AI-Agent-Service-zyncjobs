"""Token logger — tracks token usage per brain/intent for observability and cost allocation."""
import logging
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional
from recruitment_ai.brains.shared.brain_result import BrainResult

logger = logging.getLogger("tokens")


class TokenLogger:
    def __init__(self):
        self._session_totals: dict[str, dict] = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    def log(self, brain_name: str, intent: str, result: BrainResult, session_id: Optional[str] = None) -> None:
        tokens = result.tokens
        if tokens <= 0:
            return
        entry = {
            "event": "token_usage",
            "brain": brain_name,
            "intent": intent,
            "tokens": tokens,
            "session_id": session_id or "anon",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(json.dumps(entry))
        sid = session_id or "anon"
        self._session_totals[sid]["total_tokens"] += tokens
        self._session_totals[sid]["prompt_tokens"] += tokens
        self._session_totals[sid]["completion_tokens"] += tokens

    def get_session_totals(self, session_id: str) -> dict:
        return dict(self._session_totals.get(session_id, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}))

    def reset_session(self, session_id: str) -> None:
        self._session_totals.pop(session_id, None)

    def log_total_prompt(self, brain_name: str, intent: str, token_count: int, session_id: Optional[str] = None) -> None:
        logger.info(json.dumps({
            "event": "prompt_tokens",
            "brain": brain_name,
            "intent": intent,
            "prompt_tokens": token_count,
            "session_id": session_id or "anon",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    def log_total_completion(self, brain_name: str, intent: str, token_count: int, session_id: Optional[str] = None) -> None:
        logger.info(json.dumps({
            "event": "completion_tokens",
            "brain": brain_name,
            "intent": intent,
            "completion_tokens": token_count,
            "session_id": session_id or "anon",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))


token_logger = TokenLogger()

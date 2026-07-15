"""Cost logger — calculates and tracks estimated LLM cost per request."""
import logging
import json
from datetime import datetime, timezone
from typing import Optional
from recruitment_ai.brains.shared.brain_result import BrainResult

logger = logging.getLogger("cost")

COST_PER_1K_TOKENS = {
    "claude-sonnet": {"input": 0.003, "output": 0.015},
    "claude-haiku": {"input": 0.00025, "output": 0.00125},
    "qwen2.5:3b": {"input": 0.0001, "output": 0.0002},
    "qwen3:8b": {"input": 0.0002, "output": 0.0004},
    "llama3.1:8b": {"input": 0.0002, "output": 0.0004},
}

DEFAULT_COST = {"input": 0.0003, "output": 0.0006}


class CostLogger:
    def estimate_cost(self, tokens: int, model: str = "qwen2.5:3b", output_tokens: Optional[int] = None) -> float:
        if tokens <= 0:
            return 0.0
        rates = COST_PER_1K_TOKENS.get(model, DEFAULT_COST)
        input_cost = (tokens / 1000) * rates["input"]
        output_cost = ((output_tokens or tokens) / 1000) * rates["output"]
        return round(input_cost + output_cost, 6)

    def log(self, brain_name: str, intent: str, result: BrainResult, model: str = "qwen2.5:3b", session_id: Optional[str] = None) -> None:
        cost = self.estimate_cost(result.tokens, model)
        logger.info(json.dumps({
            "event": "cost",
            "brain": brain_name,
            "intent": intent,
            "tokens": result.tokens,
            "cost_usd": cost,
            "model": model,
            "session_id": session_id or "anon",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    def log_estimated(self, brain_name: str, intent: str, prompt_tokens: int, completion_tokens: int, model: str, session_id: Optional[str] = None) -> None:
        rates = COST_PER_1K_TOKENS.get(model, DEFAULT_COST)
        cost = round((prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"], 6)
        logger.info(json.dumps({
            "event": "cost_estimated",
            "brain": brain_name,
            "intent": intent,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "model": model,
            "session_id": session_id or "anon",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))


cost_logger = CostLogger()

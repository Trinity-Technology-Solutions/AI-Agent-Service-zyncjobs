"""Latency logger — measures and logs execution time per node and brain."""
import logging
import json
import time
from datetime import datetime, timezone
from typing import Optional
from recruitment_ai.brains.shared.brain_state import BrainState

logger = logging.getLogger("latency")


class LatencyLogger:
    def __init__(self):
        self._node_timings: dict[str, float] = {}

    def start_node(self, node_name: str) -> None:
        self._node_timings[node_name] = time.perf_counter()

    def end_node(self, node_name: str, state: BrainState) -> float:
        start = self._node_timings.pop(node_name, None)
        if start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(json.dumps({
            "event": "node_latency",
            "node": node_name,
            "duration_ms": round(elapsed_ms, 2),
            "session_id": state.session.id or "anon",
            "intent": state.intent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))
        return elapsed_ms

    def log_brain_latency(self, brain_name: str, duration_ms: float, state: BrainState) -> None:
        logger.info(json.dumps({
            "event": "brain_latency",
            "brain": brain_name,
            "duration_ms": round(duration_ms, 2),
            "session_id": state.session.id or "anon",
            "intent": state.intent,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))

    def log_total_latency(self, duration_ms: float, state: BrainState) -> None:
        logger.info(json.dumps({
            "event": "total_latency",
            "duration_ms": round(duration_ms, 2),
            "session_id": state.session.id or "anon",
            "intent": state.intent,
            "brain": state.execution.brain or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }))


latency_logger = LatencyLogger()

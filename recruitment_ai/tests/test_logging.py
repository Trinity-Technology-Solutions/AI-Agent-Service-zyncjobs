"""Tests for the logging package."""
import pytest
from recruitment_ai.brains.shared.brain_state import BrainState
from recruitment_ai.brains.shared.brain_result import BrainResult
from recruitment_ai.logging.execution_logger import ExecutionLogger, execution_logger
from recruitment_ai.logging.token_logger import TokenLogger, token_logger
from recruitment_ai.logging.latency_logger import LatencyLogger, latency_logger
from recruitment_ai.logging.cost_logger import CostLogger, cost_logger
from recruitment_ai.logging.audit_logger import AuditLogger, audit_logger


def _make_state(session_id="s1", user_id="u1", intent="CHAT", role="candidate"):
    state = BrainState(session__id=session_id, user__id=user_id, intent=intent, user_role=role)
    state.session.id = session_id
    state.user.id = user_id
    state.user.role = role
    return state


class TestExecutionLogger:
    def test_log_start_returns_id(self):
        state = _make_state("sess1")
        eid = execution_logger.log_start(state)
        assert eid is not None
        assert "sess1" in eid

    def test_log_end_no_error(self):
        state = _make_state()
        result = BrainResult(success=True, response={"reply": "hi"})
        execution_logger.log_end(state, result, "test_1")

    def test_log_error(self):
        state = _make_state()
        execution_logger.log_error(state, "something broke", "test_2")


class TestTokenLogger:
    def test_log_token_usage(self):
        result = BrainResult(success=True, tokens=150)
        token_logger.log("test_brain", "CHAT", result, "sess1")
        totals = token_logger.get_session_totals("sess1")
        assert totals["total_tokens"] >= 150

    def test_reset_session(self):
        token_logger.reset_session("reset_sess")
        totals = token_logger.get_session_totals("reset_sess")
        assert totals["total_tokens"] == 0

    def test_log_prompt(self):
        token_logger.log_total_prompt("test_brain", "CHAT", 100, "sess1")

    def test_log_completion(self):
        token_logger.log_total_completion("test_brain", "CHAT", 50, "sess1")


class TestLatencyLogger:
    def test_start_end_node(self):
        state = _make_state()
        latency_logger.start_node("test_node")
        import time
        time.sleep(0.001)
        elapsed = latency_logger.end_node("test_node", state)
        assert elapsed > 0

    def test_end_node_missing(self):
        state = BrainState()
        elapsed = latency_logger.end_node("nonexistent", state)
        assert elapsed == 0.0

    def test_log_brain_latency(self):
        state = _make_state()
        latency_logger.log_brain_latency("test_brain", 100.5, state)

    def test_log_total_latency(self):
        state = _make_state()
        latency_logger.log_total_latency(250.0, state)


class TestCostLogger:
    def test_estimate_cost_zero_tokens(self):
        cost = cost_logger.estimate_cost(0, "qwen2.5:3b")
        assert cost == 0.0

    def test_estimate_cost_qwen(self):
        cost = cost_logger.estimate_cost(1000, "qwen2.5:3b")
        assert cost > 0

    def test_estimate_cost_unknown_model(self):
        cost = cost_logger.estimate_cost(1000, "unknown-model")
        assert cost > 0

    def test_log_cost(self):
        result = BrainResult(success=True, tokens=500)
        cost_logger.log("test_brain", "CHAT", result, "qwen2.5:3b", "sess1")

    def test_log_estimated(self):
        cost_logger.log_estimated("test_brain", "CHAT", 100, 50, "qwen2.5:3b", "sess1")


class TestAuditLogger:
    def test_log_request(self):
        state = _make_state()
        audit_logger.log_request(state)

    def test_log_response(self):
        state = _make_state()
        result = BrainResult(success=True, response={"reply": "hello"})
        audit_logger.log_response(state, result)

    def test_log_error(self):
        state = _make_state()
        audit_logger.log_error(state, "test error")

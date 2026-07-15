"""Master Brain — orchestrates every AI request through the platform.
Architecture doc: IntentClassifier → BrainRouter → Selected Brain → LLM Service.

New in Phase 5:
  - Uses BrainRegistry instead of old build_registry() → BrainRouter.
  - Brain.run() returns BrainResult; MasterBrain merges it into BrainState.

Phase 7b:
  - Logs execution, tokens, latency, cost, and audit events.
"""
import logging
import time
from recruitment_ai.brains.base import BrainState, BrainResult
from recruitment_ai.brains.master.intent_classifier import intent_classifier
from recruitment_ai.brains.master.router import router
from recruitment_ai.logging.execution_logger import execution_logger
from recruitment_ai.logging.token_logger import token_logger
from recruitment_ai.logging.latency_logger import latency_logger
from recruitment_ai.logging.cost_logger import cost_logger
from recruitment_ai.logging.audit_logger import audit_logger

logger = logging.getLogger(__name__)


class MasterBrain:
    """Routes every request: classify intent → select brain → execute → merge result."""

    @property
    def brains(self) -> dict:
        return router.all_brains

    async def execute(self, state: BrainState) -> BrainState:
        state = await intent_classifier.classify(state)
        intent = state.intent or "CHAT"
        brain = router.get(intent)

        execution_id = execution_logger.log_start(state)
        audit_logger.log_request(state)
        start_time = time.perf_counter()

        if not brain:
            state.error = f"No brain found for intent: {intent}"
            state.result = {"error": state.error}
            execution_logger.log_error(state, state.error, execution_id)
            audit_logger.log_error(state, state.error)
            return state

        if not await brain.validate(state):
            state.error = f"Invalid input for {brain.name}"
            state.result = {"error": state.error}
            execution_logger.log_error(state, state.error, execution_id)
            audit_logger.log_error(state, state.error)
            return state

        try:
            result: BrainResult = await brain.run(state)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            state.execution.brain = brain.name
            state.execution.duration_ms = elapsed_ms
            state.execution.cache_hit = state.metadata.get("cache_hit", False)

            if result.success:
                state.result = result.data
            else:
                state.error = result.error
                state.result = {"error": result.error} if result.error else None
            state.metadata.update(result.metadata)
            state.metadata["brain"] = brain.name

            execution_logger.log_end(state, result, execution_id)
            latency_logger.log_brain_latency(brain.name, elapsed_ms, state)
            latency_logger.log_total_latency(elapsed_ms, state)
            token_logger.log(brain.name, intent, result, state.session.id)
            cost_logger.log(brain.name, intent, result, state.provider_info.model or state.model or "qwen2.5:3b", state.session.id)
            audit_logger.log_response(state, result)

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            state.execution.brain = brain.name
            state.execution.duration_ms = elapsed_ms
            logger.exception("Brain %s failed", brain.name)
            state.error = f"{brain.name} error: {str(e)}"
            state.result = {"error": state.error}
            execution_logger.log_error(state, str(e), execution_id)
            audit_logger.log_error(state, str(e))

        return state


master_brain = MasterBrain()

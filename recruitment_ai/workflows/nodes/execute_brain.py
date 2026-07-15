"""Execute brain node — delegates to MasterBrain. Skipped on cache hit."""
import logging
from recruitment_ai.brains.base import BrainState
from recruitment_ai.brains.master.master_brain import master_brain
from recruitment_ai.logging.latency_logger import latency_logger

logger = logging.getLogger(__name__)


async def execute_brain_node(state: BrainState) -> BrainState:
    latency_logger.start_node("execute_brain")
    if state.metadata.get("cache_hit"):
        latency_logger.end_node("execute_brain", state)
        return state
    state = await master_brain.execute(state)
    latency_logger.end_node("execute_brain", state)
    logger.debug("Brain executed: intent=%s error=%s", state.intent, state.error)
    return state

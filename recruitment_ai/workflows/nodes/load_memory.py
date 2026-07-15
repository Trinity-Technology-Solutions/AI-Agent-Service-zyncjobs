"""Load memory node — injects previous conversation turns into BrainState.
Uses MemoryManager which orchestrates Redis + DB conversation history
and user preferences.
"""
import logging
from recruitment_ai.brains.base import BrainState
from recruitment_ai.memory.memory_manager import memory_manager

logger = logging.getLogger(__name__)


async def load_memory_node(state: BrainState) -> BrainState:
    if not state.session_id:
        return state
    state = await memory_manager.load_memory(state)
    logger.debug("Memory loaded: %d turns for session %s", state.metadata.get("history_turns", 0), state.session_id)
    return state

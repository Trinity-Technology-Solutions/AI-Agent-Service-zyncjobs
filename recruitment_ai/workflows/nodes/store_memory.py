"""Store memory node — persists result to Redis cache + conversation history.
Cache: all successful non-chat results.
Memory: only conversational intents with a session_id (via MemoryManager).
"""
import logging
from recruitment_ai.brains.base import BrainState
from recruitment_ai.services.cache_service import cache_service
from recruitment_ai.memory.memory_manager import memory_manager

logger = logging.getLogger(__name__)


async def store_memory_node(state: BrainState) -> BrainState:
    result = state.result
    query = (state.query or "").strip()
    intent = state.intent or ""
    is_cache_hit = state.metadata.get("cache_hit", False)

    # Cache successful non-cached results
    if result and query and intent and not state.error and not is_cache_hit:
        await cache_service.set(intent, query, result)

    # Memory: conversational intents with a session only
    if state.session_id and not state.error:
        await memory_manager.store_memory(state)

    return state

"""Planner node — checks Redis cache after intent is known.
Cache hit → result is populated, execute_brain is skipped.
"""
import logging
from recruitment_ai.brains.base import BrainState
from recruitment_ai.services.cache_service import cache_service

logger = logging.getLogger(__name__)


async def planner_node(state: BrainState) -> BrainState:
    intent = state.intent or "CHAT"
    query = (state.query or "").strip()

    state.metadata["planned_brain"] = intent

    if query and intent:
        cached = await cache_service.get(intent, query)
        if cached:
            state.result = cached
            state.metadata["cache_hit"] = True
            logger.debug("Cache HIT: intent=%s", intent)

    return state

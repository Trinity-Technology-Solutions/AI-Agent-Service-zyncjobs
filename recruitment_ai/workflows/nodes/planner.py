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

    # Personalized intents must NOT share cache across users
    PERSONAL_INTENTS = {"CAREER_ADVICE", "SKILL_GAP", "CAREER_ROADMAP", "SKILL_ASSESSMENT", "INTERVIEW_PREP", "ATS_SCORE"}

    if query and intent:
        user_id = state.user_id or state.user.id or ""
        cache_query = f"{user_id}:{query}" if intent in PERSONAL_INTENTS and user_id else query
        cached = await cache_service.get(intent, cache_query)
        if cached:
            state.result = cached
            state.metadata["cache_hit"] = True
            logger.debug("Cache HIT: intent=%s", intent)

    return state

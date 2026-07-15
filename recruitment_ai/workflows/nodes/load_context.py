"""Load context node — injects user, resume, job, company, assessment data into BrainState.
Called once per request, after authentication.
Every brain receives the same context without loading data itself.
"""
import logging
from recruitment_ai.brains.base import BrainState
from recruitment_ai.context.context_manager import context_manager

logger = logging.getLogger(__name__)


async def load_context_node(state: BrainState) -> BrainState:
    if not state.user_id:
        return state
    state = await context_manager.load_context(state)
    logger.debug("Context loaded for user %s", state.user_id)
    return state

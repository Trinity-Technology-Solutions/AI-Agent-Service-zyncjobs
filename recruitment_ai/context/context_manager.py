"""ContextManager — orchestrates all context loading into BrainState.
Called once per request, after authentication, before brain execution.

Every brain receives the same context automatically.
No brain needs to fetch its own data.
"""
import logging
from recruitment_ai.brains.shared import BrainState
from recruitment_ai.context.user_context import user_context
from recruitment_ai.context.resume_context import resume_context
from recruitment_ai.context.job_context import job_context
from recruitment_ai.context.company_context import company_context
from recruitment_ai.context.assessment_context import assessment_context

logger = logging.getLogger(__name__)


class ContextManager:
    """Loads all context types into BrainState. Each loader is independent and gracefully degrades."""

    async def load_context(self, state: BrainState) -> BrainState:
        if not state.user_id:
            return state

        state = await user_context.load(state)
        state = await resume_context.load(state)
        state = await job_context.load(state)
        state = await company_context.load(state)
        state = await assessment_context.load(state)

        logger.debug("Context loaded for user %s", state.user_id)
        return state


context_manager = ContextManager()

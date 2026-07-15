"""Job context loader — fetches job description data and populates BrainState.
"""
import logging
from typing import Optional
from recruitment_ai.brains.shared import BrainState, JobContext

logger = logging.getLogger(__name__)


class JobContextLoader:
    """Loads job description / target role from state or database."""

    async def load(self, state: BrainState) -> BrainState:
        job_title = state.context.get("job_title")
        job_description = state.context.get("job_description")
        company_name = state.context.get("company")

        if not any([job_title, job_description, company_name]):
            return state

        state.context_data.job = JobContext(
            title=job_title,
            description=job_description,
            company_name=company_name,
            skills=state.context.get("skills", []),
        )

        logger.debug("Job context loaded: %s at %s", job_title, company_name)
        return state


job_context = JobContextLoader()

"""Company context loader — fetches company profile via repository and populates BrainState."""
import logging
from typing import Optional
from recruitment_ai.brains.shared import BrainState, CompanyContext
from recruitment_ai.repositories import company_repo

logger = logging.getLogger(__name__)


class CompanyContextLoader:
    async def load(self, state: BrainState) -> BrainState:
        company_name = state.context_data.job.company_name or state.context.get("company")
        if not company_name:
            return state

        profile = await company_repo.get_profile(company_name)
        if profile:
            state.context_data.company = CompanyContext(
                name=profile.get("name", company_name),
                industry=profile.get("industry"),
            )
            logger.debug("Company context loaded: %s", company_name)
        else:
            state.context_data.company = CompanyContext(name=company_name)

        return state


company_context = CompanyContextLoader()

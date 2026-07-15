"""Assessment context loader — fetches skill assessment data via repository and populates BrainState."""
import logging
from recruitment_ai.brains.shared import BrainState
from recruitment_ai.repositories import assessment_repo

logger = logging.getLogger(__name__)


class AssessmentContextLoader:
    async def load(self, state: BrainState) -> BrainState:
        user_id = state.user_id
        if not user_id:
            return state

        last_assessment = await assessment_repo.get_last_assessment_data(user_id)
        if last_assessment:
            state.context["last_assessment"] = last_assessment
            state.context_data.user_preferences["last_assessment"] = last_assessment
            logger.debug("Assessment context loaded for user %s", user_id)

        return state


assessment_context = AssessmentContextLoader()

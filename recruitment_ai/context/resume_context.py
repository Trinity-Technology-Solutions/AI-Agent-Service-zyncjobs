"""Resume context loader — fetches parsed resume data via repository and populates BrainState."""
import logging
from recruitment_ai.brains.shared import BrainState, ResumeContext
from recruitment_ai.repositories import resume_repo

logger = logging.getLogger(__name__)


class ResumeContextLoader:
    async def load(self, state: BrainState) -> BrainState:
        user_id = state.user_id
        if not user_id:
            return state

        resume_data = await resume_repo.get_resume_data(user_id)
        if resume_data:
            resume = ResumeContext(
                parsed=resume_data.get("parsed"),
                skills=resume_data.get("skills", []),
                experience=resume_data.get("experience", []),
                education=resume_data.get("education", []),
            )
            state.context_data.resume = resume
            state.context["resume"] = resume_data.get("parsed")
            state.context["current_skills"] = resume_data.get("skills", [])
            state.context["skills"] = resume_data.get("skills", [])

            logger.debug("Resume context loaded: %d skills, %d experiences",
                         len(resume.skills), len(resume.experience))

        return state


resume_context = ResumeContextLoader()

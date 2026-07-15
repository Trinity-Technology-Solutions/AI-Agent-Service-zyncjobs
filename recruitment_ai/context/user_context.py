"""User context loader — fetches user profile from DB and populates BrainState."""
import logging
from typing import Optional
from recruitment_ai.brains.shared import BrainState, UserContext
from recruitment_ai.repositories import user_repo

logger = logging.getLogger(__name__)


class UserContextLoader:
    async def load(self, state: BrainState) -> BrainState:
        user_id = state.user_id
        if not user_id:
            return state

        profile = await user_repo.get_profile(user_id)
        if profile:
            state.user = UserContext(
                id=user_id,
                email=profile.get("email"),
                role=profile.get("role", "candidate"),
                name=profile.get("name"),
            )
            state.user_role = profile.get("role", "candidate")
            state.context_data.user_preferences = profile.get("preferences", {})
            state.context["user_name"] = profile.get("name")
            state.context["current_role"] = profile.get("title")
            state.context["experience_years"] = profile.get("experience_years")
            state.context["location"] = profile.get("location")
            state.context["ats_score"] = profile.get("ats_score")
            state.context["applications_count"] = profile.get("applications_count")
            state.context["missing_skills"] = profile.get("missing_skills", [])
            state.context["skills"] = profile.get("skills", [])

            logger.debug("User context loaded: %s (%s)", user_id, profile.get("role"))
        else:
            logger.debug("No profile found for user: %s", user_id)

        return state


user_context = UserContextLoader()

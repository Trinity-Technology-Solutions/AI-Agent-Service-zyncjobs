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
            # Merge DB profile into user_preferences — do NOT overwrite what the
            # frontend already sent (systemPrompt, skills, history, etc.)
            db_prefs = profile.get("preferences", {})
            merged = {
                "user_name": profile.get("name"),
                "current_role": profile.get("title"),
                "experience_years": profile.get("experience_years"),
                "location": profile.get("location"),
                "ats_score": profile.get("ats_score"),
                "applications_count": profile.get("applications_count"),
                "missing_skills": profile.get("missing_skills", []),
                "skills": profile.get("skills", []),
                **db_prefs,
                # Frontend-sent values take priority — applied last
                **state.context_data.user_preferences,
            }
            state.context_data.user_preferences = merged
            logger.debug("User context loaded: %s (%s)", user_id, profile.get("role"))
        else:
            logger.debug("No profile found for user: %s", user_id)

        return state


user_context = UserContextLoader()

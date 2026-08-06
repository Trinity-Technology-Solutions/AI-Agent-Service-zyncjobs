"""Recruiter context loader — fetches the full employer context bundle from the Node backend.

Bundle contains: company profile, active jobs (full JD), top ranked candidates per job
(match score, ATS score, missing skills, resume summary) and pipeline stats.

Stored on: state.context_data.recruiter_context (extra key on ContextData).
"""
import logging
from recruitment_ai.brains.shared import BrainState
from recruitment_ai.services.backend_client import backend_client

logger = logging.getLogger(__name__)


class RecruiterContextLoader:
    async def load(self, state: BrainState) -> BrainState:
        if state.user_role not in ("employer", "recruiter"):
            return state

        prefs = state.context_data.user_preferences or {}
        profile = prefs.get("user_profile") or {}
        email = (state.user.email or "").strip()
        if not email:
            email = (profile.get("email") or "").strip()
        if not email:
            logger.debug("Recruiter context skipped: no employer email resolvable")
            return state

        try:
            payload = await backend_client.get_recruiter_context(email)
            if payload and isinstance(payload, dict) and (payload.get("jobs") or payload.get("company") or payload.get("stats")):
                state.context_data.recruiter_context = payload
                logger.info(
                    "Recruiter context loaded for %s: %d jobs, %d applications",
                    email, len(payload.get("jobs") or []), (payload.get("stats") or {}).get("applications", 0),
                )
            else:
                logger.debug("Recruiter context empty for %s", email)
        except Exception as e:
            logger.warning("Recruiter context load failed for %s: %s", email, e)

        return state


recruiter_context_loader = RecruiterContextLoader()

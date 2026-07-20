"""Router — maps intent strings to brain instances via BrainRegistry.
Architecture doc: BrainRouter → BrainRegistry.

Each brain registers itself at import time. Importing this module
triggers all registrations via the singleton instances below.
"""
from recruitment_ai.brains.base import BrainRegistry

router = BrainRegistry()

# ── Import brain singletons (triggers registrations) ──────────────────────
from recruitment_ai.brains.chatbot.chatbot_brain import chatbot_brain
from recruitment_ai.brains.candidate.ats_brain import ats_brain
from recruitment_ai.brains.candidate.career_brain import career_brain
from recruitment_ai.brains.candidate.career_roadmap_brain import career_roadmap_brain
from recruitment_ai.brains.candidate.cover_letter_brain import cover_letter_brain
from recruitment_ai.brains.candidate.job_matching_brain import job_matching_brain
from recruitment_ai.brains.candidate.resume_edit_brain import resume_edit_brain
from recruitment_ai.brains.candidate.resume_parser_brain import resume_parser_brain
from recruitment_ai.brains.candidate.skill_gap_brain import skill_gap_brain
from recruitment_ai.brains.employer.jd_generator_brain import jd_generator_brain
from recruitment_ai.brains.employer.job_parser_brain import job_parser_brain
from recruitment_ai.brains.employer.recruiter_brain import recruiter_brain

# ── Register brains ───────────────────────────────────────────────────────
router.register("CHAT", chatbot_brain)
router.register("JOB_PARSER", job_parser_brain)
router.register("JD_GENERATOR", jd_generator_brain)
router.register("RESUME_PARSER", resume_parser_brain)
router.register_many(
    ["RESUME_EDIT", "RESUME_BUILDER"],
    resume_edit_brain,
)
router.register("ATS_SCORE", ats_brain)
router.register("JOB_MATCH", job_matching_brain)
router.register_many(
    ["CAREER_ADVICE", "SKILL_ASSESSMENT", "INTERVIEW_PREP", "RESUME_BUILDER", "ASSESSMENT_MENTOR"],
    career_brain,
)
router.register("CAREER_ROADMAP", career_roadmap_brain)
router.register("SKILL_GAP", skill_gap_brain)
router.register("COVER_LETTER", cover_letter_brain)
router.register_many(
    ["RECRUITER", "RECRUITER_SEARCH", "RECRUITER_SHORTLIST"],
    recruiter_brain,
)

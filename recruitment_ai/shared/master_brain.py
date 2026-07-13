"""Master Brain - orchestrates all specialized brains."""
from typing import Optional
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.intent_classifier import intent_classifier
from recruitment_ai.shared.ollama_service import ollama_service
from recruitment_ai.brains.chatbot.chatbot_brain import ChatbotBrain
from recruitment_ai.brains.employer.job_parser_brain import JobParserBrain
from recruitment_ai.brains.employer.jd_generator_brain import JDGeneratorBrain
from recruitment_ai.brains.candidate.resume_parser_brain import ResumeParserBrain
from recruitment_ai.brains.candidate.ats_brain import ATSBrain
from recruitment_ai.brains.candidate.job_matching_brain import JobMatchingBrain
from recruitment_ai.brains.candidate.career_brain import CareerBrain
from recruitment_ai.brains.candidate.skill_gap_brain import SkillGapBrain
from recruitment_ai.brains.candidate.career_roadmap_brain import CareerRoadmapBrain
from recruitment_ai.brains.candidate.resume_edit_brain import resume_edit_brain
from recruitment_ai.brains.candidate.cover_letter_brain import CoverLetterBrain
from recruitment_ai.brains.employer.recruiter_brain import RecruiterBrain


class MasterBrain:
    """Master brain that routes requests to specialized brains."""

    def __init__(self):
        self.brains: dict[str, Brain] = {}
        self._initialize_brains()

    def _initialize_brains(self):
        self.brains = {
            "JOB_PARSER": JobParserBrain(),
            "JD_GENERATOR": JDGeneratorBrain(),
            "RESUME_PARSER": ResumeParserBrain(),
            "ATS_SCORE": ATSBrain(),
            "JOB_MATCH": JobMatchingBrain(),
            "CAREER_ADVICE": CareerBrain(),
            "SKILL_ASSESSMENT": CareerBrain(),
            "SKILL_GAP": SkillGapBrain(),
            "CAREER_ROADMAP": CareerRoadmapBrain(),
            "INTERVIEW_PREP": CareerBrain(),
            "RESUME_BUILDER": CareerBrain(),
            "RECRUITER": RecruiterBrain(),
            "RECRUITER_SEARCH": RecruiterBrain(),
            "RECRUITER_SHORTLIST": RecruiterBrain(),
            "RESUME_EDIT": resume_edit_brain,
            "COVER_LETTER": CoverLetterBrain(),
            "CHAT": ChatbotBrain(),
        }

    async def execute(self, state: BrainState) -> BrainState:
        if not await self._authenticate(state):
            state.error = "Authentication failed"
            return state

        state = await self._load_memory(state)
        state = await intent_classifier.classify(state)

        intent = state.intent or "CHAT"
        brain = self.brains.get(intent)

        if not brain:
            state.error = f"No brain found for intent: {intent}"
            return state

        if not await brain.validate_input(state):
            state.error = f"Invalid input for {brain.name}"
            return state

        try:
            state = await brain.run(state)
            state = await brain.post_process(state)
        except Exception as e:
            state = await brain.handle_error(state, e)

        state = await self._store_memory(state)
        return state

    async def _authenticate(self, state: BrainState) -> bool:
        return True

    async def _load_memory(self, state: BrainState) -> BrainState:
        return state

    async def _store_memory(self, state: BrainState) -> BrainState:
        return state


master_brain = MasterBrain()
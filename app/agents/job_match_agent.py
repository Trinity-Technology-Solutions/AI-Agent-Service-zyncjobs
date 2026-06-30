from typing import Optional
from .base_agent import BaseAgent
from app.tools.ats_tool import ATSTool
from app.tools.skill_extractor import SkillExtractorTool
from app.prompts.prompt_manager import prompt_manager
from app.prompts.system_prompt import JOB_MATCH_SYSTEM_PROMPT
from app.memory.memory_manager import memory


class JobMatchAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="job_match_agent",
            description="Matches resumes to job descriptions and provides match scores",
        )
        self.ats = ATSTool()
        self.skill_extractor = SkillExtractorTool()

    def system_prompt(self) -> str:
        return JOB_MATCH_SYSTEM_PROMPT

    async def execute(self, query: str, user_id: Optional[str] = None, **kwargs) -> dict:
        resume_text = kwargs.get("resume_text", "")
        job_description = kwargs.get("job_description", "")

        if not resume_text and user_id:
            resume_text = memory.get_resume(user_id).get("raw", "")

        ats_result = self.ats.run(resume_text, job_description)

        resume_skills = self.skill_extractor.run(resume_text)
        jd_skills = self.skill_extractor.run(job_description)
        missing = self.skill_extractor.suggest_missing(resume_skills, jd_skills)

        history = memory.get_history(user_id) if user_id else []
        prompt = prompt_manager.build_job_match_prompt(resume_text, job_description, ats_result)
        if history:
            prompt += "\n\nConversation history:\n" + "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in history[-5:]
            )
        prompt = self.augment_with_context(prompt, query)
        suggestions = await self.generate(prompt, system=JOB_MATCH_SYSTEM_PROMPT)

        if user_id:
            memory.store_message(user_id, "user", query)
            memory.store_message(user_id, "assistant", suggestions[:200])
            memory.store_skills(user_id, missing)

        return {
            "match_score": ats_result["score"],
            "matching_skills": ats_result.get("matching_keywords", []),
            "missing_skills": missing or ats_result.get("missing_keywords", []),
            "suggestions": suggestions,
        }

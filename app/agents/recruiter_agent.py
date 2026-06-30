from typing import Optional
from .base_agent import BaseAgent
from app.prompts.prompt_manager import prompt_manager
from app.prompts.system_prompt import JD_SYSTEM_PROMPT
from app.tools.skill_extractor import SkillExtractorTool
from app.tools.keyword_tool import KeywordTool
from app.memory.memory_manager import memory


class RecruiterAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="recruiter_agent",
            description="Generates job descriptions and helps recruiters",
        )
        self.skill_extractor = SkillExtractorTool()
        self.keyword_tool = KeywordTool()

    def system_prompt(self) -> str:
        return JD_SYSTEM_PROMPT

    async def execute(self, query: str, user_id: Optional[str] = None, **kwargs) -> dict:
        title = kwargs.get("title", "")
        experience = kwargs.get("experience_level", "")
        skills = kwargs.get("skills", [])

        detected = self.skill_extractor.run(title + " " + query)
        all_skills = list(dict.fromkeys(list(skills) + detected))

        history = memory.get_history(user_id) if user_id else []
        prompt = prompt_manager.build_jd_prompt(title, experience, all_skills, query)
        if history:
            prompt += "\n\nConversation history:\n" + "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in history[-5:]
            )
        prompt = self.augment_with_context(prompt, query)
        jd = await self.generate(prompt, system=JD_SYSTEM_PROMPT)

        if user_id:
            memory.store_message(user_id, "user", query)
            memory.store_message(user_id, "assistant", jd[:200])

        return {
            "job_description": jd,
            "suggested_skills": all_skills,
            "keywords": self.keyword_tool.run(title + " " + " ".join(all_skills), 10),
        }

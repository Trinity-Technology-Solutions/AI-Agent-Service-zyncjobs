from typing import Optional
from .base_agent import BaseAgent
from app.prompts.prompt_manager import prompt_manager
from app.prompts.system_prompt import INTERVIEW_SYSTEM_PROMPT
from app.tools.skill_extractor import SkillExtractorTool
from app.tools.keyword_tool import KeywordTool
from app.memory.memory_manager import memory


class InterviewAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="interview_agent",
            description="Generates interview questions for any role",
        )
        self.skill_extractor = SkillExtractorTool()
        self.keyword_tool = KeywordTool()

    def system_prompt(self) -> str:
        return INTERVIEW_SYSTEM_PROMPT

    async def execute(self, query: str, user_id: Optional[str] = None, **kwargs) -> dict:
        job_title = kwargs.get("job_title", "")
        skills = kwargs.get("skills", [])
        level = kwargs.get("experience_level", "mid")

        if not skills and user_id:
            skills = memory.get_skills(user_id)

        detected = self.skill_extractor.run(job_title + " " + query)
        all_skills = list(dict.fromkeys(list(skills) + detected))

        history = memory.get_history(user_id) if user_id else []
        prompt = prompt_manager.build_interview_prompt(
            job_title, all_skills, level,
            f"{query}\nKey areas: {self.keyword_tool.run(job_title, 5)}"
        )
        if history:
            prompt += "\n\nConversation history:\n" + "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in history[-5:]
            )
        prompt = self.augment_with_context(prompt, query)
        questions = await self.generate(prompt, system=INTERVIEW_SYSTEM_PROMPT)

        if user_id:
            memory.store_message(user_id, "user", query)
            memory.store_message(user_id, "assistant", questions[:200])

        return {"questions": questions}

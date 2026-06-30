from typing import Optional
from .base_agent import BaseAgent
from app.prompts.prompt_manager import prompt_manager
from app.prompts.system_prompt import RESUME_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT, SKILLS_SYSTEM_PROMPT
from app.tools.resume_parser import ResumeParserTool
from app.tools.ats_tool import ATSTool
from app.tools.grammar_tool import GrammarTool
from app.tools.skill_extractor import SkillExtractorTool
from app.tools.summary_tool import SummaryTool
from app.memory.memory_manager import memory
from app.utils.logger import logger


class ResumeAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="resume_agent",
            description="Handles resume improvement, ATS scoring, summaries, skill suggestions, and cover letters",
        )
        self.parser = ResumeParserTool()
        self.ats = ATSTool()
        self.grammar = GrammarTool()
        self.skill_extractor = SkillExtractorTool()
        self.summary_tool = SummaryTool()

    def system_prompt(self) -> str:
        return RESUME_SYSTEM_PROMPT

    async def execute(self, query: str, user_id: Optional[str] = None, **kwargs) -> dict:
        resume_text = kwargs.get("resume_text", "")
        job_description = kwargs.get("job_description", "")

        if not resume_text and user_id:
            resume_text = memory.get_resume(user_id).get("raw", "")

        parsed = self.parser.run(resume_text)
        grammar_issues = self.grammar.run(resume_text)

        ats_result = {}
        if job_description:
            ats_result = self.ats.run(resume_text, job_description)

        detected_skills = self.skill_extractor.extract_from_resume(parsed)

        history = memory.get_history(user_id) if user_id else []
        improve_prompt = prompt_manager.build_resume_prompt(
            resume_text, parsed, grammar_issues, ats_result or None, detected_skills
        )
        if history:
            improve_prompt += "\n\nConversation history:\n" + "\n".join(
                f"{m['role']}: {m['content'][:200]}" for m in history[-5:]
            )
        improve_prompt = self.augment_with_context(improve_prompt, query)
        improved = await self.generate(improve_prompt, system=RESUME_SYSTEM_PROMPT)

        summary = self.summary_tool.run(parsed)
        if not summary or len(summary.split()) < 15:
            summary_prompt = self.augment_with_context(
                prompt_manager.build_summary_prompt(resume_text), query
            )
            summary = await self.generate(summary_prompt, system=SUMMARY_SYSTEM_PROMPT)

        skills_prompt = self.augment_with_context(
            prompt_manager.build_skills_prompt(resume_text, detected_skills), query
        )
        skills_raw = await self.generate(skills_prompt, system=SKILLS_SYSTEM_PROMPT)
        skills_list = list(dict.fromkeys(s.strip() for s in skills_raw.split(",") if s.strip()))

        if user_id:
            memory.store_message(user_id, "user", query)
            memory.store_message(user_id, "assistant", improved[:200])
            memory.store_resume(user_id, {"raw": resume_text, "parsed": parsed, "improved": improved})
            memory.store_skills(user_id, detected_skills + skills_list)

        return {
            "improved_resume": improved,
            "ats_score": ats_result.get("score", 0) if ats_result else None,
            "summary": summary,
            "skills_suggested": skills_list[:10],
            "grammar_issues": grammar_issues,
            "detected_skills": detected_skills,
        }

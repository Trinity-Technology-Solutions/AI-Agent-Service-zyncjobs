"""Job Parser Brain — full enterprise pipeline.
Pipeline: BrainState.request → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict
from recruitment_ai.prompts import get_prompt, get_system_prompt

SKILL_KEYWORDS = [
    "python", "java", "javascript", "react", "node", "sql", "aws", "docker",
    "kubernetes", "git", "linux", "agile", "scrum", "rest", "api", "html",
    "css", "typescript", "go", "rust", "c++", "c#", ".net", "spring",
    "django", "flask", "fastapi", "postgresql", "mongodb", "redis",
]


class JobParserBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        from recruitment_ai.utils.ocr import extract_text
        raw_content = state.request.file_content or state.file_content or state.request.query or state.query or ""
        file_type = state.request.file_type or state.file_type or "txt"
        content = extract_text(raw_content, file_type) if file_type != "txt" else raw_content

        if not content.strip():
            return BrainResult(success=False, response={"error": "No job description provided"})

        system = get_system_prompt("job_parser")
        prompt = f"""{get_prompt("job_parser_prompt")}\n\nJob Description:\n{content}"""

        try:
            result = await llm_service.generate(
                brain_name="job_parser",
                prompt=prompt,
                system=system,
                temperature=0.1,
                max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(
                response=parsed,
                metadata={"parser": "llm"},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response=self._fallback_parse(content),
                metadata={"parser": "fallback", "fallback_reason": str(e)},
                execution_time=time.perf_counter() - start,
            )

    def _fallback_parse(self, content: str) -> dict:
        lines = content.split("\n")
        skills = [kw for kw in SKILL_KEYWORDS if re.search(rf"\b{kw}\b", content, re.IGNORECASE)]
        return {
            "title": self._extract_title(lines),
            "company": "", "location": "", "job_type": "full-time",
            "experience_level": "mid", "salary_min": None, "salary_max": None,
            "currency": "USD", "skills_required": skills[:10], "skills_preferred": [],
            "responsibilities": [], "requirements": [], "benefits": [],
            "description": content[:2000],
        }

    def _extract_title(self, lines: list[str]) -> str:
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 100:
                return line
        return "Software Engineer"


job_parser_brain = JobParserBrain()

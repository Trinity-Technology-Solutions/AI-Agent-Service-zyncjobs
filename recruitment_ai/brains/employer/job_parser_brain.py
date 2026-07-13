"""Job Parser Brain - parses job descriptions into structured data."""
import re
import json
from typing import Optional
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

JOB_PARSER_SYSTEM = """You are a job description parser. Extract structured information from job postings.
Return ONLY valid JSON. No extra text, no markdown, no explanation."""


JD_PARSER_PROMPT = """Extract structured information from this job description.

Return JSON with these fields:
{{
  "title": "job title",
  "company": "company name",
  "location": "city, state/country",
  "job_type": "full-time|part-time|contract|internship",
  "experience_level": "entry|mid|senior|lead|executive",
  "salary_min": number or null,
  "salary_max": number or null,
  "currency": "USD|INR|EUR|etc",
  "skills_required": ["skill1", "skill2", ...],
  "skills_preferred": ["skill1", "skill2", ...],
  "responsibilities": ["resp1", "resp2", ...],
  "requirements": ["req1", "req2", ...],
  "benefits": ["benefit1", "benefit2", ...],
  "description": "full job description text"
}}

Only return valid JSON. No extra text."""


class JobParserBrain(Brain):
    """Parses job descriptions into structured data."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        from recruitment_ai.utils.ocr import extract_text
        raw_content = state.file_content or state.query or ""
        content = extract_text(raw_content, state.file_type or "txt") if state.file_type else raw_content
        if not content.strip():
            state.error = "No job description provided"
            return state

        try:
            result = await ollama_service.generate(
                brain_name="job_parser",
                prompt=f"{JD_PARSER_PROMPT}\n\nJob Description:\n{content}",
                system=JOB_PARSER_SYSTEM,
                temperature=0.1,
                max_tokens=1024,
            )
            state.result = self._parse_json(result)
            state.metadata["parser"] = "llm"
        except Exception as e:
            state.result = self._fallback_parse(content)
            state.metadata["parser"] = "fallback"
            state.metadata["fallback_reason"] = str(e)

        return state

    def _parse_json(self, text: str) -> dict:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return self._empty_result()

    def _fallback_parse(self, content: str) -> dict:
        import re
        lines = content.split("\n")
        skills = []
        skill_keywords = [
            "python", "java", "javascript", "react", "node", "sql", "aws", "docker",
            "kubernetes", "git", "linux", "agile", "scrum", "rest", "api", "html",
            "css", "typescript", "go", "rust", "c++", "c#", ".net", "spring",
            "django", "flask", "fastapi", "postgresql", "mongodb", "redis"
        ]
        for kw in skill_keywords:
            if re.search(rf"\b{kw}\b", content, re.IGNORECASE):
                skills.append(kw)

        return {
            "title": self._extract_title(lines),
            "company": "",
            "location": "",
            "job_type": "full-time",
            "experience_level": "mid",
            "salary_min": None,
            "salary_max": None,
            "currency": "USD",
            "skills_required": skills[:10],
            "skills_preferred": [],
            "responsibilities": [],
            "requirements": [],
            "benefits": [],
            "description": content[:2000],
        }

    def _extract_title(self, lines: list[str]) -> str:
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 100:
                return line
        return "Software Engineer"

    def _empty_result(self) -> dict:
        return {
            "title": "", "company": "", "location": "", "job_type": "full-time",
            "experience_level": "mid", "salary_min": None, "salary_max": None,
            "currency": "USD", "skills_required": [], "skills_preferred": [],
            "responsibilities": [], "requirements": [], "benefits": [], "description": "",
        }


job_parser_brain = JobParserBrain()
"""Resume Parser Brain - parses resumes into structured data."""
import re
import json
from typing import Optional
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

RESUME_PARSER_SYSTEM = """You are a resume parser. Extract structured information from resumes.
Return ONLY valid JSON. No extra text, no markdown, no explanation."""


RESUME_PARSER_PROMPT = """Parse this resume into structured JSON.

Return JSON with:
{{
  "personal_info": {{
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "linkedin": "",
    "github": "",
    "portfolio": ""
  }},
  "summary": "",
  "experience": [
    {{
      "company": "",
      "title": "",
      "location": "",
      "start_date": "",
      "end_date": "",
      "current": false,
      "description": "",
      "technologies": []
    }}
  ],
  "education": [
    {{
      "degree": "",
      "institution": "",
      "location": "",
      "graduation_year": 0,
      "gpa": 0.0
    }}
  ],
  "skills": {{
    "technical": [],
    "soft": [],
    "languages": [],
    "tools": [],
    "frameworks": []
  }},
  "projects": [
    {{
      "name": "",
      "description": "",
      "technologies": [],
      "url": ""
    }}
  ],
  "certifications": [
    {{
      "name": "",
      "issuer": "",
      "year": 0
    }}
  ],
  "languages": []
}}

Only return valid JSON. No extra text."""


class ResumeParserBrain(Brain):
    """Parses resumes into structured data."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        content = state.file_content or state.query or ""
        if not content.strip():
            state.error = "No resume content provided"
            return state

        try:
            result = await ollama_service.generate(
                brain_name="resume_parser",
                prompt=f"{RESUME_PARSER_PROMPT}\n\nResume:\n{content}",
                system=RESUME_PARSER_SYSTEM,
                temperature=0.1,
                max_tokens=2048,
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
        return {
            "personal_info": self._extract_personal_info(content),
            "summary": self._extract_summary(content),
            "experience": self._extract_experience(content),
            "education": self._extract_education(content),
            "skills": self._extract_skills(content),
            "projects": [],
            "certifications": [],
            "languages": [],
        }

    def _extract_personal_info(self, text: str) -> dict:
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        phone_match = re.search(r"[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}", text)
        linkedin_match = re.search(r"linkedin\.com/in/[\w-]+", text, re.IGNORECASE)
        github_match = re.search(r"github\.com/[\w-]+", text, re.IGNORECASE)

        return {
            "name": "",
            "email": email_match.group() if email_match else "",
            "phone": phone_match.group() if phone_match else "",
            "location": "",
            "linkedin": linkedin_match.group() if linkedin_match else "",
            "github": github_match.group() if github_match else "",
            "portfolio": "",
        }

    def _extract_summary(self, text: str) -> str:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "summary" in line.lower() or "objective" in line.lower() or "profile" in line.lower():
                return " ".join(lines[i+1:i+4]).strip()[:500]
        return lines[0] if lines else ""

    def _extract_experience(self, text: str) -> list:
        return []

    def _extract_education(self, text: str) -> list:
        return []

    def _extract_skills(self, text: str) -> dict:
        skill_categories = {
            "technical": ["python", "java", "javascript", "react", "node", "sql", "aws", "docker", "kubernetes"],
            "soft": ["communication", "leadership", "teamwork", "problem solving"],
            "languages": ["english", "spanish", "hindi", "french"],
            "tools": ["git", "jira", "jenkins", "docker", "kubernetes"],
            "frameworks": ["django", "flask", "fastapi", "spring", "express", "nextjs"],
        }
        found = {cat: [] for cat in skill_categories}
        text_lower = text.lower()
        for cat, skills in skill_categories.items():
            for skill in skills:
                if skill in text_lower:
                    found[cat].append(skill)
        return found

    def _empty_result(self) -> dict:
        return {
            "personal_info": {"name": "", "email": "", "phone": "", "location": "", "linkedin": "", "github": "", "portfolio": ""},
            "summary": "", "experience": [], "education": [], "skills": {"technical": [], "soft": [], "languages": [], "tools": [], "frameworks": []},
            "projects": [], "certifications": [], "languages": [],
        }


resume_parser_brain = ResumeParserBrain()
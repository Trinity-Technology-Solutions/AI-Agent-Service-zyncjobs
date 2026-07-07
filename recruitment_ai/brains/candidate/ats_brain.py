"""ATS Brain - calculates ATS score for resumes against job descriptions."""
import re
import json
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

ATS_SYSTEM = """You are an ATS (Applicant Tracking System) analyzer.
Analyze resumes against job descriptions and return ONLY valid JSON. No extra text, no markdown, no explanation."""


ATS_PROMPT = """Analyze this resume against the job description and return ATS score.

Resume:
{resume}

Job Description:
{job_description}

Return JSON with:
{{
  "ats_score": 0-100,
  "keyword_match": {{
    "matched": ["skill1", "skill2"],
    "missing": ["skill3", "skill4"],
    "match_percentage": 0-100
  }},
  "formatting_score": 0-100,
  "section_completeness": 0-100,
  "experience_relevance": 0-100,
  "suggestions": ["Add missing skill: X", "Use standard headings", "Quantify achievements"],
  "passes_ats": true/false
}}

Only return valid JSON."""


class ATSBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        context = state.context or {}
        resume = context.get("resume", state.query or "")
        job_description = context.get("job_description", "")

        if not resume.strip():
            state.error = "No resume provided"
            return state

        prompt = ATS_PROMPT.format(resume=resume[:3000], job_description=job_description[:3000])

        try:
            result = await ollama_service.generate(
                brain_name="ats_scanner",
                prompt=prompt,
                system=ATS_SYSTEM,
                temperature=0.1,
                max_tokens=1024,
            )
            state.result = self._parse_json(result)
            state.metadata["model"] = "qwen3:8b"
        except Exception as e:
            state.result = self._rule_based_ats(resume, job_description)
            state.metadata["model"] = "rule_based"
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

    def _rule_based_ats(self, resume: str, jd: str) -> dict:
        resume_lower = resume.lower()
        jd_lower = jd.lower()

        skills = ["python", "java", "javascript", "react", "node", "sql", "aws", "docker", "kubernetes", "git", "linux", "agile", "scrum", "rest", "api", "html", "css", "typescript", "go", "rust", "c++", "c#", ".net", "spring", "django", "flask", "fastapi", "postgresql", "mongodb", "redis"]
        
        jd_skills = set(re.findall(rf"\b({'|'.join(skills)})\b", jd_lower))
        resume_skills = set(re.findall(rf"\b({'|'.join(skills)})\b", resume_lower))

        matched = list(jd_skills & resume_skills)
        missing = list(jd_skills - resume_skills)
        match_pct = int(len(matched) / len(jd_skills) * 100) if jd_skills else 100

        has_sections = all(s in resume_lower for s in ["experience", "education", "skills"])
        formatting = 80 if has_sections else 50
        completeness = 90 if len(resume) > 1000 else 60
        exp_relevance = 70 if matched else 30

        ats_score = int(match_pct * 0.4 + formatting * 0.2 + completeness * 0.2 + exp_relevance * 0.2)

        suggestions = []
        if missing:
            suggestions.append(f"Add missing keywords: {', '.join(missing[:5])}")
        if not has_sections:
            suggestions.append("Use standard section headings (Experience, Education, Skills)")
        if len(resume) < 1000:
            suggestions.append("Expand resume with more details")
        suggestions.append("Quantify achievements with numbers and percentages")

        return {
            "ats_score": ats_score,
            "keyword_match": {"matched": matched, "missing": missing, "match_percentage": match_pct},
            "formatting_score": formatting,
            "section_completeness": completeness,
            "experience_relevance": exp_relevance,
            "suggestions": suggestions,
            "passes_ats": ats_score >= 70,
        }

    def _empty_result(self) -> dict:
        return {
            "ats_score": 0,
            "keyword_match": {"matched": [], "missing": [], "match_percentage": 0},
            "formatting_score": 0,
            "section_completeness": 0,
            "experience_relevance": 0,
            "suggestions": ["Provide resume and job description"],
            "passes_ats": False,
        }


ats_brain = ATSBrain()
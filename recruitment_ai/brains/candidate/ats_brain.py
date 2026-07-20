"""ATS Brain — full enterprise pipeline.
Pipeline: BrainState.context_data (resume + job) → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict
from recruitment_ai.prompts import get_prompt, get_system_prompt
from recruitment_ai.brains.candidate.skill_keywords import SKILL_KEYWORDS, extract_matched_skills


class ATSBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        resume = state.context_data.resume.parsed or state.context.get("resume") or state.request.query or state.query or ""
        job_description = state.context_data.job.description or state.context.get("job_description", "")

        resume_text = json.dumps(resume) if isinstance(resume, dict) else str(resume)
        if not resume_text.strip():
            return BrainResult(success=False, response={"error": "No resume provided"})

        prompt = get_prompt("ats_prompt", resume=resume_text[:3000], job_description=job_description[:3000])
        system = get_system_prompt("ats")

        try:
            result = await llm_service.generate(
                brain_name="ats_scanner",
                prompt=prompt,
                system=system,
                temperature=0.1,
                max_tokens=512,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(
                response=parsed,
                metadata={"model": "llm"},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response=self._rule_based_ats(resume_text, job_description),
                metadata={"model": "rule_based", "fallback_reason": str(e)},
                warnings=["LLM analysis failed, used rule-based fallback"],
                execution_time=time.perf_counter() - start,
            )

    def _rule_based_ats(self, resume: str, jd: str) -> dict:
        resume_skills = extract_matched_skills(resume)
        jd_skills = extract_matched_skills(jd)
        matched = list(jd_skills & resume_skills)
        missing = list(jd_skills - resume_skills)
        n_jd = len(jd_skills)
        if n_jd == 0:
            match_pct = 0  # nothing to match against
        else:
            match_pct = round(len(matched) / n_jd * 100)
        has_sections = all(s in resume.lower() for s in ["experience", "education", "skills"])
        formatting = 80 if has_sections else 50
        completeness = 90 if len(resume) > 1000 else 60
        total_resume_skills = len(resume_skills)
        exp_relevance = (
            80 if total_resume_skills >= 10 else
            70 if total_resume_skills >= 5 else
            50 if total_resume_skills >= 2 else
            30
        )
        ats_score = round(match_pct * 0.4 + formatting * 0.2 + completeness * 0.2 + exp_relevance * 0.2)
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


ats_brain = ATSBrain()

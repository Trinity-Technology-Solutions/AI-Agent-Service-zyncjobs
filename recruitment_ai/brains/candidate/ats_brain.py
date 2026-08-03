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
from recruitment_ai.brains.candidate.skill_keywords import SKILL_VARIATIONS, SKILL_KEYWORDS as DEFAULT_SKILL_KEYWORDS


class ATSBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        resume = state.context_data.resume.parsed or state.context.get("resume") or state.request.query or state.query or ""
        job_description = state.context_data.job.description or state.context.get("job_description", "")

        resume_text = json.dumps(resume) if isinstance(resume, dict) else str(resume)
        job_text = str(job_description)
        
        # Early validation - no resume
        if not resume_text.strip():
            return BrainResult(success=False, response={"error": "No resume provided"})
        
        if not job_text.strip():
            return BrainResult(success=False, response={"error": "Both resume and job description required"})

        prompt = get_prompt("ats_prompt", resume=resume_text[:3000], job_description=job_text[:3000])
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
            
            # Ensure we have valid parsed response; if not, fall back to rule-based
            if self._is_valid_parsed_result(parsed):
                response = parsed
                model_used = "llm"
            else:
                response = self._rule_based_ats(resume_text, job_text)
                model_used = "rule_based"
                
            return BrainResult(
                response=response,
                metadata={"model": model_used},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response=self._rule_based_ats(resume_text, job_text),
                metadata={"model": "rule_based", "fallback_reason": str(e)},
                warnings=["LLM analysis failed, used rule-based fallback"],
                execution_time=time.perf_counter() - start,
            )

    def _is_valid_parsed_result(self, parsed: any) -> bool:
        """Check if parsed result is valid and contains expected fields."""
        if not parsed or not isinstance(parsed, dict):
            return False
        if parsed.get("error"):
            return False
        
        # Check for essential ATS fields
        essential_fields = ["ats_score", "keyword_match", "formatting_score", "section_completeness", "experience_relevance", "suggestions"]
        has_essential = all(field in parsed for field in essential_fields)
        
        # Ensure keyword_match structure
        if "keyword_match" in parsed:
            km = parsed["keyword_match"]
            if not isinstance(km.get("matched"), list) or not isinstance(km.get("missing"), list):
                return False
        
        # Validate score ranges
        if not isinstance(parsed.get("ats_score"), (int, float)) or not (0 <= parsed.get("ats_score", 0) <= 100):
            return False
            
        return has_essential

    def _rule_based_ats(self, resume: str, jd: str) -> dict:
        resume_skills = extract_matched_skills(resume)
        jd_skills = extract_matched_skills(jd)
        matched = list(jd_skills & resume_skills)
        missing = list(jd_skills - resume_skills)
        n_jd = len(jd_skills)
        match_pct = round(len(matched) / n_jd * 100) if n_jd > 0 else 0

        resume_lower = resume.lower()
        resume_words = len(resume.split())

        # ── Section detection ────────────────────────────────────────────
        has_experience = any(h in resume_lower for h in ["experience", "work experience", "employment", "professional experience"])
        has_education = any(h in resume_lower for h in ["education", "university", "college", "degree", "academic"])
        has_skills = any(h in resume_lower for h in ["skills", "technical skills", "competencies", "expertise"])
        has_summary = any(h in resume_lower for h in ["summary", "profile", "objective", "about"])
        section_count = sum([has_experience, has_education, has_skills, has_summary])

        # ── Formatting: strict — must have real sections with content ────
        if section_count >= 4:
            formatting = 80
        elif section_count == 3:
            formatting = 65
        elif section_count == 2:
            formatting = 45
        elif section_count == 1:
            formatting = 25
        else:
            formatting = 10

        # ── Completeness: penalise thin resumes heavily ──────────────────
        if resume_words >= 500 and section_count >= 3:
            completeness = 80
        elif resume_words >= 300 and section_count >= 2:
            completeness = 60
        elif resume_words >= 150:
            completeness = 40
        elif resume_words >= 50:
            completeness = 20
        else:
            completeness = 5

        # ── Experience relevance: skill count + match quality ────────────
        total_skills = len(resume_skills)
        if total_skills >= 10 and match_pct >= 50:
            exp_relevance = 80
        elif total_skills >= 6 and match_pct >= 30:
            exp_relevance = 65
        elif total_skills >= 3:
            exp_relevance = 45
        elif total_skills >= 1:
            exp_relevance = 30
        else:
            exp_relevance = 10

        # ── ATS score: keyword match is the primary driver ───────────────
        ats_score = round(
            match_pct * 0.50 +
            formatting * 0.20 +
            completeness * 0.15 +
            exp_relevance * 0.15
        )
        ats_score = max(0, min(100, ats_score))

        suggestions = []
        if missing:
            suggestions.append(f"Add missing keywords: {', '.join(missing[:5])}")
        if not has_experience:
            suggestions.append("Add a clear Experience section with job titles and bullet points")
        if not has_education:
            suggestions.append("Add an Education section with degree and institution")
        if not has_skills:
            suggestions.append("Add a Skills section with specific technical skills")
        if not has_summary:
            suggestions.append("Add a Professional Summary at the top")
        if resume_words < 300:
            suggestions.append("Resume is too short — expand with detailed bullet points and achievements")
        if match_pct < 40 and n_jd > 0:
            suggestions.append("Low keyword match — tailor your skills and experience to the job description")
        suggestions.append("Quantify achievements with numbers, percentages, and measurable outcomes")

        return {
            "ats_score": ats_score,
            "keyword_match": {"matched": matched, "missing": missing, "match_percentage": match_pct},
            "formatting_score": formatting,
            "section_completeness": completeness,
            "experience_relevance": exp_relevance,
            "suggestions": suggestions,
            "passes_ats": ats_score >= 65,
        }


ats_brain = ATSBrain()

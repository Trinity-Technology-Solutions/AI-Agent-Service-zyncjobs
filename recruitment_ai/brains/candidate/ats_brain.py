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
        if n_jd == 0:
            match_pct = 0  # nothing to match against
        else:
            match_pct = round(len(matched) / n_jd * 100)
            
        # Enhanced section detection - look for actual section headers
        resume_lower = resume.lower()
        has_experience_section = any(header in resume_lower for header in ["experience", "work experience", "employment history", "professional experience"])
        has_education_section = any(header in resume_lower for header in ["education", "academic background", "academic history", "degrees", "university", "college"])
        has_skills_section = any(header in resume_lower for header in ["skills", "technical skills", "skill set", "competencies", "abilities", "expertise"])
        has_sections = has_experience_section and has_education_section and has_skills_section
        
        # Calculate formatting score based on actual section headers
        if has_sections:
            formatting = 85  # Good score for having all standard sections
        elif has_experience_section and has_education_section:
            formatting = 70  # Partial score - missing skills section
        elif has_experience_section or has_education_section or has_skills_section:
            formatting = 55  # Minimal score
        else:
            formatting = 30  # Very poor formatting
        
        # Enhanced resume completeness assessment
        resume_words = len(resume.split())
        if resume_words > 1000:
            completeness = 90
        elif resume_words > 600:
            completeness = 85
        elif resume_words > 300:
            completeness = 75
        elif resume_words > 100:
            completeness = 60
        elif resume_words > 50:
            completeness = 50
        else:
            completeness = 30
        
        # Better experience relevance - consider multiple factors
        total_resume_skills = len(resume_skills)
        # Weight skills more heavily than previous implementation
        if total_resume_skills >= 12:
            exp_relevance = 88
        elif total_resume_skills >= 8:
            exp_relevance = 80
        elif total_resume_skills >= 5:
            exp_relevance = 70
        elif total_resume_skills >= 2:
            exp_relevance = 60
        elif total_resume_skills >= 1:
            exp_relevance = 50
        elif resume_words > 500:
            exp_relevance = 45  # Low skills but substantial content
        else:
            exp_relevance = 35
        
        # Enhanced ATS score calculation - give proper weight to match percentage
        ats_score = round(match_pct * 0.5 + formatting * 0.2 + completeness * 0.15 + exp_relevance * 0.15)
        suggestions = []
        
        if missing:
            suggestions.append(f"Add missing keywords: {', '.join(missing[:5])}")
        if has_sections:
            suggestions.append("Resume has all standard sections (Excellent!)")
        else:
            if not has_experience_section:
                suggestions.append("Add a clear Experience/Worked section")
            if not has_education_section:
                suggestions.append("Add Education section with degrees and institutions")
            if not has_skills_section:
                suggestions.append("Add a comprehensive Skills section listing technical and soft skills")
        if len(resume) < 500:
            suggestions.append("Add more detailed descriptions (expand resume)")
        elif len(resume) < 1000:
            suggestions.append("Expand bullet points with quantifiable achievements")
        suggestions.append("Quantify achievements with numbers and percentages")
        
        # Ensure passes_ats is reasonable
        passes_ats = ats_score >= 70
        
        return {
            "ats_score": ats_score,
            "keyword_match": {"matched": matched, "missing": missing, "match_percentage": match_pct},
            "formatting_score": formatting,
            "section_completeness": completeness,
            "experience_relevance": exp_relevance,
            "suggestions": suggestions,
            "passes_ats": passes_ats,
        }


ats_brain = ATSBrain()

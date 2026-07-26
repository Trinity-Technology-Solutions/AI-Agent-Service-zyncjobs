"""Job Matching Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict
from recruitment_ai.brains.candidate.skill_keywords import SKILL_KEYWORDS, extract_matched_skills
from recruitment_ai.brains.candidate.skill_keywords import SKILL_VARIATIONS, SKILL_KEYWORDS as DEFAULT_SKILL_KEYWORDS


JOB_MATCH_SYSTEM = """You are a job matching expert. Evaluate candidate profiles against job requirements.
Return ONLY valid JSON. No extra text, no markdown, no explanation."""

JOB_MATCH_PROMPT = """Match candidate profile to job requirements.

Candidate Profile:
{candidate_profile}

Job Requirements:
{job_requirements}

Return JSON with:
{{
  "match_score": 0-100,
  "skill_match": {{
    "required_matched": ["skill1", "skill2"],
    "required_missing": ["skill3"],
    "preferred_matched": ["skill4"],
    "match_percentage": 0-100
  }},
  "experience_match": {{
    "years_required": 0,
    "years_candidate": 0,
    "level_match": "exact|close|gap"
  }},
  "location_match": "exact|remote|relocate|mismatch",
  "salary_match": "within_range|above|below|unknown",
  "strengths": ["strength1", "strength2"],
  "gaps": ["gap1", "gap2"],
  "recommendation": "strong_match|good_match|potential_match|poor_match"
}}

Only return valid JSON."""

class JobMatchingBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        candidate = state.context_data.resume.parsed or state.context.get("candidate_profile", state.request.query or state.query or "")
        job = state.context_data.job.description or state.context.get("job_requirements", "")

        candidate_text = json.dumps(candidate) if isinstance(candidate, dict) else str(candidate)
        job_text = str(job)

        if not candidate_text.strip() or not job_text.strip():
            return BrainResult(success=False, response={"error": "Both candidate profile and job requirements required"})

        prompt = JOB_MATCH_PROMPT.format(candidate_profile=candidate_text[:2000], job_requirements=job_text[:2000])

        try:
            result = await llm_service.generate(
                brain_name="job_matching",
                prompt=prompt,
                system=JOB_MATCH_SYSTEM,
                temperature=0.1,
                max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            
            # Ensure we have valid parsed response; if not, fall back to rule-based
            if self._is_valid_parsed_result(parsed):
                response = parsed
                model_used = "llm"
            else:
                response = self._rule_based_match(candidate_text, job_text)
                model_used = "rule_based"
                
            return BrainResult(response=response, metadata={"model": model_used}, execution_time=time.perf_counter() - start)
        except Exception as e:
            return BrainResult(
                response=self._rule_based_match(candidate_text, job_text),
                metadata={"model": "rule_based", "fallback_reason": str(e)},
                execution_time=time.perf_counter() - start,
            )

    def _is_valid_parsed_result(self, parsed: any) -> bool:
        """Check if parsed result is valid and contains expected fields."""
        if not parsed or not isinstance(parsed, dict):
            return False
        if parsed.get("error"):
            return False
        
        # Check for essential match fields
        essential_fields = ["match_score", "skill_match", "experience_match", "recommendation"]
        return all(field in parsed for field in essential_fields)

    def _rule_based_match(self, candidate: str, job: str) -> dict:
        cand_skills = extract_matched_skills(candidate)
        job_skills = extract_matched_skills(job)
        required = job_skills
        matched = list(required & cand_skills)
        missing = list(required - cand_skills)
        match_pct = round(len(matched) / len(required) * 100) if required else 0
        
        # Enhanced experience matching
        exp_match = re.search(r"(\d+)\+?\s*years?", job.lower())
        years_req = int(exp_match.group(1)) if exp_match else 0
        
        # Calculate experience score
        if years_req == 0:
            exp_score = 85  # No specific requirement - pass
        elif years_req <= 2:
            exp_score = max(0, 75 - years_req * 5)  # Some deduction for recent requirement
        elif years_req <= 5:
            exp_score = max(0, 60 - years_req * 5)  # Moderate deduction
        else:
            exp_score = max(0, 40 - years_req * 3)  # Higher deduction for senior roles
            
        # Calculate total score with better weighting
        score = round(match_pct * 0.65 + exp_score * 0.35)

        if score >= 80:
            rec = "strong_match"
        elif score >= 65:
            rec = "good_match"
        elif score >= 45:
            rec = "potential_match"
        else:
            rec = "poor_match"

        return {
            "match_score": int(score),
            "skill_match": {
                "required_matched": matched, "required_missing": missing,
                "preferred_matched": [], "match_percentage": match_pct,
            },
            "experience_match": {"years_required": years_req, "years_candidate": 0, "level_match": "exact" if years_req == 0 else "gap"},
            "location_match": "remote", "salary_match": "unknown",
            "strengths": matched[:5], "gaps": missing[:5], "recommendation": rec,
        }


job_matching_brain = JobMatchingBrain()

"""Job Matching Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict

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
            return BrainResult(response=parsed, metadata={"model": "llm"}, execution_time=time.perf_counter() - start)
        except Exception as e:
            return BrainResult(
                response=self._rule_based_match(candidate_text, job_text),
                metadata={"model": "rule_based", "fallback_reason": str(e)},
                execution_time=time.perf_counter() - start,
            )

    def _rule_based_match(self, candidate: str, job: str) -> dict:
        cand_lower = candidate.lower()
        job_lower = job.lower()
        all_skills = ["python", "java", "javascript", "react", "node", "sql", "aws", "docker", "kubernetes", "git", "linux", "agile", "scrum", "rest", "api", "html", "css", "typescript", "go", "rust", "c++", "c#", ".net", "spring", "django", "flask", "fastapi", "postgresql", "mongodb", "redis"]
        cand_skills = set(s for s in all_skills if s in cand_lower)
        job_skills = set(s for s in all_skills if s in job_lower)
        required = job_skills
        matched = list(required & cand_skills)
        missing = list(required - cand_skills)
        match_pct = int(len(matched) / len(required) * 100) if required else 100
        exp_match = re.search(r"(\d+)\+?\s*years?", job_lower)
        years_req = int(exp_match.group(1)) if exp_match else 0
        score = match_pct * 0.6 + (100 if years_req == 0 else min(100, 100 - years_req * 10)) * 0.4

        if score >= 80:
            rec = "strong_match"
        elif score >= 60:
            rec = "good_match"
        elif score >= 40:
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

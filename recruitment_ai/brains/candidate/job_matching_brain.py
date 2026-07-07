"""Job Matching Brain - matches candidates to jobs based on skills."""
import re
import json
from typing import Optional
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

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
    """Matches candidates to jobs."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        context = state.context or {}
        candidate = context.get("candidate_profile", state.query or "")
        job = context.get("job_requirements", "")

        if not candidate.strip() or not job.strip():
            state.error = "Both candidate profile and job requirements required"
            return state

        prompt = JOB_MATCH_PROMPT.format(candidate_profile=candidate[:2000], job_requirements=job[:2000])

        try:
            result = await ollama_service.generate(
                brain_name="job_matching",
                prompt=prompt,
                system=JOB_MATCH_SYSTEM,
                temperature=0.1,
                max_tokens=1024,
            )
            state.result = self._parse_json(result)
            state.metadata["model"] = "qwen3:8b"
        except Exception as e:
            state.result = self._rule_based_match(candidate, job)
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

    def _rule_based_match(self, candidate: str, job: str) -> dict:
        import re
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
                "required_matched": matched,
                "required_missing": missing,
                "preferred_matched": [],
                "match_percentage": match_pct,
            },
            "experience_match": {
                "years_required": years_req,
                "years_candidate": 0,
                "level_match": "exact" if years_req == 0 else "gap",
            },
            "location_match": "remote",
            "salary_match": "unknown",
            "strengths": matched[:5],
            "gaps": missing[:5],
            "recommendation": rec,
        }

    def _empty_result(self) -> dict:
        return {
            "match_score": 0,
            "skill_match": {"required_matched": [], "required_missing": [], "preferred_matched": [], "match_percentage": 0},
            "experience_match": {"years_required": 0, "years_candidate": 0, "level_match": "gap"},
            "location_match": "mismatch",
            "salary_match": "unknown",
            "strengths": [], "gaps": [], "recommendation": "poor_match",
        }


job_matching_brain = JobMatchingBrain()
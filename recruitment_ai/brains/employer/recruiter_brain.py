"""Recruiter Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict
from recruitment_ai.prompts import get_prompt, get_system_prompt


class RecruiterBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        query = state.request.query or state.query or ""
        ctx = state.context_data

        has_structured_context = bool(ctx.job.title or ctx.job.description or ctx.company.name or state.context.get("candidates"))
        if not has_structured_context:
            return await self._chat_recruiter(query, state.context_data.user_preferences.get("history", []), start)

        if "shortlist" in query.lower() or "evaluate" in query.lower():
            return await self._shortlist(state.context, start)
        return await self._search(query, state.context, start)

    async def _chat_recruiter(self, query: str, history: list, start: float) -> BrainResult:
        clean_query = re.sub(r'^recruiter:\s*', '', query, flags=re.IGNORECASE).strip()
        history_text = ""
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role}: {content}\n"
        prompt = clean_query
        if history_text:
            prompt = f"Previous conversation:\n{history_text}\n\nRecruiter: {clean_query}"
        system = get_prompt("recruiter_chat_system")
        try:
            reply = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=0.4, max_tokens=600,
            )
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            return BrainResult(
                response={"reply": reply, "intent": "RECRUITER"},
                execution_time=time.perf_counter() - start,
            )
        except Exception:
            return BrainResult(
                response={"reply": "I'm having trouble right now. Please try again.", "intent": "RECRUITER"},
            )

    async def _search(self, query: str, context: dict, start: float) -> BrainResult:
        prompt = get_prompt("recruiter_prompt",
            query=query or "Find candidates",
            filters=json.dumps(context.get("filters", {})),
            skills=", ".join(context.get("skills", [])),
            experience_level=context.get("experience_level", "mid"),
            location=context.get("location", ""),
        )
        system = get_system_prompt("recruiter")
        try:
            result = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=0.3, max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
        except Exception as e:
            return BrainResult(
                response=self._fallback_search(context),
                metadata={"fallback": True, "error": str(e)},
            )

    async def _shortlist(self, context: dict, start: float) -> BrainResult:
        job = context.get("job_requirements", "")
        candidates = context.get("candidates", [])
        prompt = SHORTLIST_PROMPT.format(job_requirements=job[:2000], candidates=json.dumps(candidates)[:3000])
        system = get_system_prompt("recruiter")
        try:
            result = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=0.1, max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
        except Exception as e:
            return BrainResult(
                response={"shortlisted": [], "top_candidate_id": "", "summary": "Shortlisting evaluation unavailable"},
                metadata={"error": str(e)},
            )

    def _fallback_search(self, context: dict) -> dict:
        skills = context.get("skills", ["Python", "JavaScript"])
        return {
            "search_strategy": f"Search candidates with skills: {', '.join(skills)}",
            "recommended_filters": {
                "skills": skills, "experience": context.get("experience_level", "mid"),
                "location": context.get("location", "remote"),
            },
            "screening_questions": [
                f"Can you describe your experience with {skills[0] if skills else 'your primary skill'}?",
                "What project are you most proud of?",
            ],
            "evaluation_criteria": {"skill_weight": 40, "experience_weight": 30, "education_weight": 15, "location_weight": 10, "other_weight": 5},
            "interview_suggestions": {"rounds": 3, "topics": skills[:3], "estimated_duration_minutes": 60},
            "advice": "Focus on practical skills assessment rather than years of experience.",
        }


SHORTLIST_PROMPT = """Evaluate candidates for shortlisting.

Job Requirements: {job_requirements}
Candidates: {candidates}

Return JSON with:
{{"shortlisted": [{{"candidate_id": "", "name": "", "match_score": 0, "strengths": [], "gaps": [], "recommendation": "strong|consider|reject"}}],
  "top_candidate_id": "",
  "summary": "Brief evaluation summary"
}}"""


recruiter_brain = RecruiterBrain()

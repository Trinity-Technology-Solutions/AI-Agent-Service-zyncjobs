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

        has_structured_context = bool(
            ctx.job.title or ctx.job.description or ctx.company.name
            or state.context.get("candidates") or state.context.get("candidate") or state.context.get("job")
        )
        if not has_structured_context:
            return await self._chat_recruiter(state, query, start)

        if "shortlist" in query.lower() or "evaluate" in query.lower():
            return await self._shortlist(state.context, start)
        if "score" in query.lower() or "rank" in query.lower():
            return await self._score_candidate(state, start)
        return await self._search(query, state.context, start)

    def _build_employer_context(self, prefs: dict) -> str:
        """Build employer context block from frontend-sent jobs_context and user_profile."""
        parts = []
        profile = prefs.get("user_profile") or {}
        if isinstance(profile, dict):
            name = profile.get("name") or profile.get("fullName") or ""
            company = profile.get("company") or profile.get("companyName") or ""
            if name:
                parts.append(f"Recruiter Name: {name}")
            if company:
                parts.append(f"Company: {company}")
        jobs = prefs.get("jobs_context") or []
        if isinstance(jobs, list) and jobs:
            job_lines = []
            for j in jobs[:5]:
                if not isinstance(j, dict):
                    continue
                title = j.get("jobTitle") or j.get("title") or ""
                skills = j.get("skills")
                skills_text = ", ".join(skills[:5]) if isinstance(skills, list) else (str(skills) if skills else "")
                loc = j.get("location") or ""
                exp = j.get("experienceRange") or j.get("experienceLevel") or ""
                line = f"- {title} (location: {loc}, experience: {exp}".rstrip(", ")
                if skills_text:
                    line += f", skills: {skills_text}"
                job_lines.append(line + ")")
            if job_lines:
                parts.append("Employer Active Jobs:\n" + "\n".join(job_lines))
        return "\n".join(parts)

    async def _chat_recruiter(self, state: BrainState, query: str, start: float) -> BrainResult:
        prefs = state.context_data.user_preferences or {}
        clean_query = re.sub(r'^recruiter:\s*', '', query, flags=re.IGNORECASE).strip()
        history_text = ""
        for turn in (prefs.get("history", []) or [])[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role}: {content}\n"

        # Frontend systemPrompt override wins (it already embeds job context)
        system = prefs.get("systemPrompt") or get_system_prompt("recruiter_chat")
        employer_context = self._build_employer_context(prefs)
        if "{{ employer_context }}" in system:
            system = system.replace(
                "{{ employer_context }}",
                f"Employer Context:\n{employer_context}" if employer_context
                else "Employer Context: (No active jobs available yet.)",
            )

        prompt = clean_query
        if history_text:
            prompt = f"Previous conversation:\n{history_text}\n\nRecruiter: {clean_query}"
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

    async def _score_candidate(self, state: BrainState, start: float) -> BrainResult:
        prefs = state.context_data.user_preferences
        context = state.context
        job_data = context.get("job") or prefs.get("job") or {}
        job_desc = job_data.get("description") or job_data.get("jobDescription") or state.context_data.job.description or ""
        candidate = context.get("candidate") or context.get("candidates", [{}])[0] if context.get("candidates") else prefs.get("candidate") or {}
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else {}
        if not candidate or not candidate.get("resume"):
            candidate = {"resume": context.get("candidate_resume", "") or prefs.get("candidate_resume", "")}
        prompt = f"""Score this candidate for the job.

Job: {job_desc[:2000]}
Candidate: {json.dumps(candidate)[:2000]}

Return JSON with:
{{"overallScore": 0-100, "skillsScore": 0-100, "experienceScore": 0-100,
  "shouldReject": false, "reasons": [], "feedback": "",
  "recommendation": "strong|consider|reject",
  "matchingSkills": [], "missingSkills": [],
  "summary": "", "breakdown": {{"skill_weight": 0, "experience_weight": 0, "education_weight": 0, "location_weight": 0}}
}}"""
        system = get_system_prompt("recruiter")
        try:
            result = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=0.1, max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            if "overallScore" in parsed or "score" in parsed:
                return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
            return BrainResult(
                response={"overallScore": 50, "skillsScore": 50, "reasons": ["AI evaluation unavailable"]},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response={"overallScore": 50, "skillsScore": 50, "reasons": [str(e)]},
                metadata={"fallback": True, "error": str(e)},
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

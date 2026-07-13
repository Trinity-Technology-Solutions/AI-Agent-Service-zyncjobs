"""Recruiter Assistant Brain - candidate search, filtering, shortlisting."""
import re
import json
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

RECRUITER_SYSTEM = """You are an expert technical recruiter and hiring manager.
Return ONLY valid JSON as specified. No extra text, no markdown, no explanation."""

RECRUITER_CHAT_SYSTEM = """You are ZyncJobs AI Recruiter Assistant — an expert recruitment automation assistant for employers and HR teams on ZyncJobs.
Help recruiters with candidate evaluation, job postings, interview questions, screening criteria, offer letters, and hiring advice.
NEVER mention other job sites (LinkedIn, Indeed, Glassdoor, Naukri, Monster, Shine, etc.). Focus ONLY on ZyncJobs platform.
Keep responses concise, professional, and actionable. Use bullet points for lists."""

RECRUITER_PROMPT = """You are a Recruiter Assistant. Help find and evaluate candidates.

Employer Request: {query}
Filters: {filters}
Required Skills: {skills}
Experience Level: {experience_level}
Location: {location}

Return JSON with:
{{
  "search_strategy": "Best approach to find candidates",
  "recommended_filters": {{ "skills": [], "experience": "", "location": "" }},
  "screening_questions": ["Q1", "Q2"],
  "evaluation_criteria": {{
    "skill_weight": 40,
    "experience_weight": 30,
    "education_weight": 15,
    "location_weight": 10,
    "other_weight": 5
  }},
  "interview_suggestions": {{
    "rounds": 3,
    "topics": ["topic1", "topic2"],
    "estimated_duration_minutes": 60
  }},
  "advice": "Brief actionable hiring advice"
}}"""


SHORTLIST_PROMPT = """Evaluate candidates for shortlisting.

Job Requirements: {job_requirements}
Candidates: {candidates}

Return JSON with:
{{
  "shortlisted": [
    {{
      "candidate_id": "",
      "name": "",
      "match_score": 0-100,
      "strengths": [],
      "gaps": [],
      "recommendation": "strong|consider|reject"
    }}
  ],
  "top_candidate_id": "",
  "summary": "Brief evaluation summary"
}}"""


class RecruiterBrain(Brain):
    """Recruiter Assistant - candidate search and evaluation."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        context = state.context or {}
        query = state.query or ""

        # Free-text chat query — return conversational reply
        has_structured_context = any(k in context for k in ("job_requirements", "candidates", "filters", "skills"))
        if not has_structured_context:
            return await self._chat_recruiter(state, query)

        if "shortlist" in query.lower() or "evaluate" in query.lower():
            return await self._shortlist(state, context)
        return await self._search(state, context)

    async def _chat_recruiter(self, state: BrainState, query: str) -> BrainState:
        """Handle free-text recruiter questions conversationally."""
        clean_query = re.sub(r'^recruiter:\s*', '', query, flags=re.IGNORECASE).strip()
        try:
            reply = await ollama_service.generate(
                brain_name="recruiter",
                prompt=clean_query,
                system=RECRUITER_CHAT_SYSTEM,
                temperature=0.4,
                max_tokens=600,
            )
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            state.result = {"reply": reply, "intent": "RECRUITER"}
        except Exception:
            state.result = {"reply": "I'm having trouble right now. Please try again.", "intent": "RECRUITER"}
        return state

    async def _search(self, state: BrainState, context: dict) -> BrainState:
        prompt = RECRUITER_PROMPT.format(
            query=state.query or "Find candidates",
            filters=json.dumps(context.get("filters", {})),
            skills=", ".join(context.get("skills", [])),
            experience_level=context.get("experience_level", "mid"),
            location=context.get("location", ""),
        )
        try:
            result = await ollama_service.generate(
                brain_name="recruiter",
                prompt=prompt,
                system=RECRUITER_SYSTEM,
                temperature=0.3,
                max_tokens=1024,
            )
            state.result = self._parse_json(result)
        except Exception as e:
            state.result = self._fallback_search(context)
            state.metadata["fallback"] = True
            state.metadata["error"] = str(e)
        return state

    async def _shortlist(self, state: BrainState, context: dict) -> BrainState:
        job = context.get("job_requirements", state.query or "")
        candidates = context.get("candidates", [])

        prompt = SHORTLIST_PROMPT.format(
            job_requirements=job[:2000],
            candidates=json.dumps(candidates)[:3000],
        )
        try:
            result = await ollama_service.generate(
                brain_name="recruiter",
                prompt=prompt,
                system=RECRUITER_SYSTEM,
                temperature=0.1,
                max_tokens=1024,
            )
            state.result = self._parse_json(result)
        except Exception as e:
            state.result = {
                "shortlisted": [],
                "top_candidate_id": "",
                "summary": "Shortlisting evaluation unavailable",
            }
            state.metadata["error"] = str(e)
        return state

    def _parse_json(self, text: str) -> dict:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _fallback_search(self, context: dict) -> dict:
        skills = context.get("skills", ["Python", "JavaScript"])
        return {
            "search_strategy": f"Search candidates with skills: {', '.join(skills)}",
            "recommended_filters": {
                "skills": skills,
                "experience": context.get("experience_level", "mid"),
                "location": context.get("location", "remote"),
            },
            "screening_questions": [
                f"Can you describe your experience with {skills[0] if skills else 'your primary skill'}?",
                "What project are you most proud of?",
            ],
            "evaluation_criteria": {
                "skill_weight": 40,
                "experience_weight": 30,
                "education_weight": 15,
                "location_weight": 10,
                "other_weight": 5,
            },
            "interview_suggestions": {
                "rounds": 3,
                "topics": skills[:3],
                "estimated_duration_minutes": 60,
            },
            "advice": "Focus on practical skills assessment rather than years of experience.",
        }


recruiter_brain = RecruiterBrain()

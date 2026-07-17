"""Career Brain — full enterprise pipeline.
All context pre-loaded by Context Manager into BrainState.
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict
from recruitment_ai.prompts import get_prompt, get_system_prompt

CAREER_SYSTEM = get_system_prompt("career")
CAREER_CHAT_SYSTEM_TPL = get_prompt("career_chat_system", user_context="{user_context}")
INTERVIEW_SYSTEM = get_system_prompt("interview")
RESUME_BUILDER_SYSTEM_TMPL = """You are a professional resume writer.
If the input contains specific technologies/skills/tools, use ONLY those. Do NOT invent or assume any.
If the input only mentions a role title (e.g. 'Backend Developer, Fresher' with no specifics), you may suggest commonly used technologies relevant to that role.
Return ONLY valid JSON as specified. No extra text, no markdown, no explanation."""
SKILL_ASSESSMENT_SYSTEM = """You are a technical interviewer generating MCQ questions.
Return ONLY valid JSON. No markdown, no explanation, no code blocks."""


class CareerBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        intent = state.intent or "CAREER_ADVICE"
        query = state.request.query or state.query or ""
        ctx = state.context_data

        if intent == "SKILL_ASSESSMENT":
            return await self._skill_assessment(state, ctx, start)
        elif intent == "INTERVIEW_PREP":
            return await self._interview_prep(state, ctx, start)
        elif intent == "RESUME_BUILDER":
            return await self._resume_builder(state, ctx, start)
        elif intent == "CAREER_ADVICE":
            if ctx.resume.skills or ctx.resume.parsed:
                return await self._career_advice(state, ctx, start)
            return await self._chat_advice(state, query, start)
        else:
            return await self._chat_advice(state, query, start)

    def _build_user_context(self, state: BrainState) -> str:
        parts = []
        resume = state.context_data.resume
        prefs = state.context_data.user_preferences
        if resume.skills:
            parts.append(f"Skills: {', '.join(resume.skills[:10])}")
        if prefs.get("ats_score"):
            parts.append(f"ATS Score: {prefs['ats_score']}%")
        if prefs.get("experience_years"):
            parts.append(f"Experience: {prefs['experience_years']}")
        return "\n".join(parts) or "No profile data available."

    async def _chat_advice(self, state: BrainState, query: str, start: float) -> BrainResult:
        clean_query = re.sub(r'^career advice:\s*', '', query, flags=re.IGNORECASE).strip()
        user_context = self._build_user_context(state)
        history = state.context_data.user_preferences.get("history", [])
        history_text = ""
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role}: {content}\n"
        rag = state.retrieved_documents.chunks or []
        rag_text = "\n".join(c.get("text", "") for c in rag[:3]) if rag else ""
        system = get_prompt("career_chat_system", user_context=user_context)
        parts = []
        if history_text:
            parts.append(f"Previous conversation:\n{history_text}")
        parts.append(f"User: {clean_query}")
        if rag_text:
            parts.append(f"Relevant context:\n{rag_text}")
        prompt = "\n\n".join(parts)
        try:
            reply = await llm_service.generate(
                brain_name="career_advice", prompt=prompt, system=system,
                temperature=0.4, max_tokens=600,
            )
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            return BrainResult(
                response={"reply": reply, "intent": "CAREER_ADVICE", "rag_used": bool(rag_text)},
                execution_time=time.perf_counter() - start,
            )
        except Exception:
            return BrainResult(
                response={"reply": "I'm having trouble right now. Please try again in a moment.", "intent": "CAREER_ADVICE"},
            )

    async def _career_advice(self, state: BrainState, ctx, start: float) -> BrainResult:
        resume = ctx.resume
        prefs = ctx.user_preferences
        prompt = CAREER_ADVICE_PROMPT.format(
            current_role=prefs.get("current_role", "Software Engineer"),
            target_role=prefs.get("target_role", state.request.query or "Senior Software Engineer"),
            current_skills=", ".join(resume.skills or ["Python", "JavaScript"]),
            experience_years=prefs.get("experience_years", 3),
            location=prefs.get("location", "Remote"),
        )
        try:
            result = await llm_service.generate(
                brain_name="career_advice", prompt=prompt, system=CAREER_SYSTEM,
                temperature=0.3, max_tokens=2048,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
        except Exception:
            return BrainResult(
                response=self._fallback_career_advice(prefs),
                metadata={"fallback": True},
            )

    async def _skill_assessment(self, state: BrainState, ctx, start: float) -> BrainResult:
        skill = ctx.user_preferences.get("skill", "Python")
        prompt = SKILL_ASSESSMENT_PROMPT.format(
            skill=skill,
            level=ctx.user_preferences.get("level", "intermediate"),
            count=ctx.user_preferences.get("count", 10),
        )
        try:
            result = await llm_service.generate(
                brain_name="skill_assessment", prompt=prompt, system=SKILL_ASSESSMENT_SYSTEM,
                temperature=0.3, max_tokens=3000,
            )
            parsed = validate_json_strict(result, "object") or {}
            questions = parsed.get("questions", [])
            valid = [q for q in questions if (
                q.get("question") and isinstance(q.get("options"), list)
                and len(q["options"]) == 4 and isinstance(q.get("correctAnswer"), int)
                and 0 <= q["correctAnswer"] <= 3
            )]
            if len(valid) < 5:
                raise ValueError(f"Only {len(valid)} valid questions generated")
            return BrainResult(
                response={"questions": valid[:10]},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response={"questions": [], "error": str(e)},
                metadata={"fallback": True},
            )

    async def _interview_prep(self, state: BrainState, ctx, start: float) -> BrainResult:
        resume = ctx.resume
        prompt = INTERVIEW_PREP_PROMPT.format(
            role=ctx.user_preferences.get("role", "Software Engineer"),
            level=ctx.user_preferences.get("level", "mid"),
            skills=", ".join(resume.skills or ["Python", "JavaScript"]),
            interview_type=ctx.user_preferences.get("interview_type", "technical"),
        )
        try:
            result = await llm_service.generate(
                brain_name="interview_prep", prompt=prompt, system=INTERVIEW_SYSTEM,
                temperature=0.3, max_tokens=2048,
            )
            parsed = validate_json_strict(result, "object") or {}
            if not parsed:
                raise ValueError("Empty result")
            return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
        except Exception:
            return BrainResult(
                response={"questions": [], "topics_to_review": [], "tips": []},
                metadata={"fallback": True},
            )

    async def _resume_builder(self, state: BrainState, ctx, start: float) -> BrainResult:
        query = state.request.query or state.query or ""
        if ctx.resume.parsed or ctx.resume.skills:
            prompt = RESUME_BUILDER_PROMPT.format(
                personal_info=json.dumps({"name": state.user.name}),
                experience=json.dumps(ctx.resume.experience),
                education=json.dumps(ctx.resume.education),
                skills=json.dumps(ctx.resume.skills),
                target_role=ctx.user_preferences.get("target_role", "Software Engineer"),
            )
        else:
            prompt = f"""Generate resume content based on this description:\n\n{query}\n\nIf the description has specific technologies/skills, use ONLY those. If it only mentions a role (e.g. 'Backend Developer, Fresher'), suggest commonly used technologies for that role.\n\nReturn JSON with:\n{{"summary": "...", "experience_bullets": [], "skills_formatted": {{}}, "ats_keywords": []}}"""
        try:
            result = await llm_service.generate(
                brain_name="resume_builder", prompt=prompt, system=RESUME_BUILDER_SYSTEM_TMPL,
                temperature=0.3, max_tokens=2048,
            )
            parsed = validate_json_strict(result, "object") or {}
            if not parsed:
                raise ValueError("Empty parse")
            return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
        except Exception:
            return BrainResult(
                response={"summary": "", "experience_bullets": [], "skills_formatted": {}, "ats_keywords": []},
            )

    def _fallback_career_advice(self, prefs: dict) -> dict:
        return {
            "career_path": [{"step": 1, "title": "Learn " + prefs.get("target_role", "target role"), "skills_to_learn": [], "estimated_months": 12}],
            "skill_gaps": ["Identify missing skills from job descriptions"],
            "recommended_courses": [], "certifications": [], "timeline_months": 12,
            "advice": "Focus on building projects that demonstrate target role skills.",
        }


CAREER_ADVICE_PROMPT = """Provide career advice for a candidate.

Current Role: {current_role}
Target Role: {target_role}
Current Skills: {current_skills}
Experience Years: {experience_years}
Location: {location}

Return JSON with:
{{
  "career_path": [
    {{"step": 1, "title": "Junior Developer", "skills_to_learn": ["skill1"], "estimated_months": 6}}
  ],
  "skill_gaps": ["gap1", "gap2"],
  "recommended_courses": [{{"title": "Course", "platform": "Coursera", "url": ""}}],
  "certifications": ["cert1"],
  "timeline_months": 18,
  "advice": "Brief actionable advice"
}}"""

SKILL_ASSESSMENT_PROMPT = """Generate exactly {count} multiple choice questions about {skill} for a {level} level assessment.

Return ONLY this JSON structure:
{{
  "questions": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correctAnswer": 0
    }}
  ]
}}

Rules:
- EXACTLY {count} questions about {skill}
- Each question must have exactly 4 real options
- correctAnswer is 0-based index
- Return ONLY the JSON object"""

INTERVIEW_PREP_PROMPT = """Generate interview preparation for a role.

Role: {role}
Level: {level}
Skills: {skills}
Interview Type: {interview_type}

Return JSON with:
{{
  "questions": [{{"question": "...", "type": "technical|behavioral", "difficulty": "easy|medium|hard", "expected_answer": "..."}}],
  "topics_to_review": ["topic1"],
  "tips": ["tip1", "tip2"]
}}"""

RESUME_BUILDER_PROMPT = """Generate resume content for a candidate.

Personal Info: {personal_info}
Experience: {experience}
Education: {education}
Skills: {skills}
Target Role: {target_role}

Return JSON with:
{{
  "summary": "Professional summary (3-4 lines)",
  "experience_bullets": [{{"company": "", "bullets": ["bullet1"]}}],
  "skills_formatted": {{"technical": [], "soft": []}},
  "ats_keywords": ["keyword1", "keyword2"]
}}"""


career_brain = CareerBrain()

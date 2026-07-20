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

        if intent == "ASSESSMENT_MENTOR":
            return await self._assessment_mentor(state, query, start)
        elif intent == "SKILL_ASSESSMENT":
            return await self._skill_assessment(state, ctx, start)
        elif intent == "INTERVIEW_PREP":
            return await self._interview_prep(state, ctx, start)
        elif intent == "RESUME_BUILDER":
            return await self._resume_builder(state, ctx, start)
        elif intent == "CAREER_ADVICE":
            has_profile = (
                ctx.resume.skills
                or ctx.user_preferences.get("skills")
                or ctx.user_preferences.get("current_role")
                or ctx.user_preferences.get("user_name")
            )
            if has_profile:
                return await self._chat_advice(state, query, start)
            return await self._chat_advice(state, query, start)
        else:
            return await self._chat_advice(state, query, start)

    def _build_user_context(self, state: BrainState) -> str:
        parts = []
        resume = state.context_data.resume
        prefs = state.context_data.user_preferences

        # Name
        name = prefs.get("user_name") or state.user.name
        if name:
            parts.append(f"Name: {name}")

        # Current role — from prefs (sent by frontend buildUserContext)
        current_role = prefs.get("current_role") or prefs.get("jobTitle")
        if current_role:
            parts.append(f"Current Role: {current_role}")

        # Target role
        target_role = prefs.get("target_role") or prefs.get("careerGoal")
        if target_role:
            parts.append(f"Target Role: {target_role}")

        # Skills — merge resume.skills + prefs skills
        skills = resume.skills or []
        pref_skills = prefs.get("skills", [])
        if isinstance(pref_skills, list):
            skills = list(dict.fromkeys(skills + pref_skills))  # deduplicate
        if skills:
            parts.append(f"Skills: {', '.join(skills[:15])}")

        # Experience
        exp = prefs.get("experience_years")
        if exp:
            parts.append(f"Experience: {exp} years")

        # ATS score
        ats = prefs.get("ats_score")
        if ats:
            parts.append(f"ATS Score: {ats}%")

        # Missing skills
        missing = prefs.get("missing_skills", [])
        if missing:
            parts.append(f"Missing Skills: {', '.join(missing[:5])}")

        # Location
        location = prefs.get("location")
        if location:
            parts.append(f"Location: {location}")

        return "\n".join(parts) if parts else "No profile data available."

    async def _chat_advice(self, state: BrainState, query: str, start: float) -> BrainResult:
        clean_query = re.sub(r'^(career advice|mentor):\s*', '', query, flags=re.IGNORECASE).strip()
        prefs = state.context_data.user_preferences
        history = prefs.get("history", [])
        history_text = ""
        for turn in history[-6:]:
            history_text += f"{turn.get('role', 'user')}: {turn.get('content', '')}\n"

        system_override = prefs.get("systemPrompt")

        if system_override:
            # Caller controls full context — skip RAG, use systemPrompt directly
            system = system_override
            parts = []
            if history_text:
                parts.append(f"Previous conversation:\n{history_text}")
            parts.append(f"User: {clean_query}")
        else:
            rag = state.retrieved_documents.chunks or []
            rag_text = "\n".join(c.get("text", "") for c in rag[:3]) if rag else ""
            user_context = self._build_user_context(state)
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
                response={"reply": reply, "intent": "CAREER_ADVICE"},
                execution_time=time.perf_counter() - start,
            )
        except Exception:
            return BrainResult(
                response={"reply": "I'm having trouble right now. Please try again in a moment.", "intent": "CAREER_ADVICE"},
            )

    async def _assessment_mentor(self, state: BrainState, query: str, start: float) -> BrainResult:
        clean_query = re.sub(r'^mentor:\s*', '', query, flags=re.IGNORECASE).strip()
        prefs = state.context_data.user_preferences

        skill = prefs.get("skill", "")
        score = prefs.get("score", 0)
        questions = prefs.get("questions", [])  # list of {num, question, options, userAnswer, correctAnswer, isCorrect}

        # Find which question number user is asking about
        num_match = re.search(
            r'\b(?:q(?:uestion)?\s*#?\s*(\d+)|(\d+)(?:st|nd|rd|th)?\s*(?:question|q\b))',
            clean_query, re.IGNORECASE
        )
        target_q = None
        if num_match:
            qnum = int(num_match.group(1) or num_match.group(2))
            # num may be int or str depending on JSON serialisation
            target_q = next(
                (q for q in questions if str(q.get("num", "")) == str(qnum)),
                None
            )

        if target_q:
            system = "You are an AI assessment mentor. Explain mistakes clearly and educationally. Never mention ZyncJobs platform features."
            prompt = (
                f"Assessment: {skill} | Score: {score}%\n\n"
                f"Question {target_q['num']}: {target_q['question']}\n"
                f"Options: {', '.join(target_q.get('options', []))}\n"
                f"Candidate answered: {target_q['userAnswer']} {'✓' if target_q.get('isCorrect') else '✗'}\n"
                f"Correct answer: {target_q['correctAnswer']}\n\n"
                f"User asks: {clean_query}\n\n"
                f"Explain:\n1. Why the candidate's answer is {'correct' if target_q.get('isCorrect') else 'incorrect'}.\n"
                f"2. Why '{target_q['correctAnswer']}' is the correct answer.\n"
                f"3. Give a short learning tip."
            )
        else:
            # General mentor question — list all wrong answers as context
            wrong = [q for q in questions if not q.get("isCorrect") and q.get("correctAnswer") != "Open ended"]
            wrong_summary = "\n".join(
                f"Q{q['num']}: {q['question']} | Your answer: {q['userAnswer']} | Correct: {q['correctAnswer']}"
                for q in wrong
            ) or "None"
            system = "You are an AI assessment mentor. Be specific and educational. Never mention ZyncJobs platform features."
            prompt = (
                f"Assessment: {skill} | Score: {score}%\n\n"
                f"Wrong answers:\n{wrong_summary}\n\n"
                f"User asks: {clean_query}"
            )

        try:
            reply = await llm_service.generate(
                brain_name="assessment_mentor", prompt=prompt, system=system,
                temperature=0.4, max_tokens=600,
            )
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            return BrainResult(
                response={"reply": reply, "intent": "ASSESSMENT_MENTOR"},
                execution_time=time.perf_counter() - start,
            )
        except Exception:
            return BrainResult(
                response={"reply": "I'm having trouble right now. Please try again.", "intent": "ASSESSMENT_MENTOR"},
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

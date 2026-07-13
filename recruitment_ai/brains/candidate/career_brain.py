"""Career Brain - handles career advice, roadmap, skill gap, interview prep, resume builder."""
import re
import json
from typing import Optional
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

CAREER_SYSTEM = """You are an expert career advisor for tech professionals.
Return ONLY valid JSON as specified. No extra text, no markdown, no explanation."""

CAREER_CHAT_SYSTEM = """You are ZyncJobs AI Career Mentor — an expert, personalized career advisor for the ZyncJobs platform.
You already know the candidate's profile. Use it to give specific, data-driven advice.
Never say generic things. Always refer to their actual skills, role, ATS score, and goals.
NEVER mention other job sites (LinkedIn, Indeed, Glassdoor, Naukri, Monster, Shine, etc.).
Direct candidates to ZyncJobs platform features only.
Be direct, encouraging, and mentor-like. Use bullet points. Max 3-4 short paragraphs.

Candidate Profile:
{user_context}"""

INTERVIEW_SYSTEM = """You are an expert technical interviewer.
Return ONLY valid JSON as specified. No extra text, no markdown, no explanation."""

RESUME_BUILDER_SYSTEM = """You are a professional resume writer.
Return ONLY valid JSON as specified. No extra text, no markdown, no explanation."""


CAREER_ADVICE_PROMPT = """Provide career advice for a candidate.

Current Role: {current_role}
Target Role: {target_role}
Current Skills: {current_skills}
Experience Years: {experience_years}
Location: {location}

Return JSON with:
{{
  "career_path": [
    {{"step": 1, "title": "Junior Developer", "skills_to_learn": ["skill1"], "estimated_months": 6}},
    {{"step": 2, "title": "Mid Developer", "skills_to_learn": ["skill2"], "estimated_months": 12}}
  ],
  "skill_gaps": ["gap1", "gap2", "gap3"],
  "recommended_courses": [
    {{"title": "Course Name", "platform": "Coursera|Udemy|edX", "url": ""}}
  ],
  "certifications": ["cert1", "cert2"],
  "timeline_months": 18,
  "advice": "Brief actionable advice"
}}"""


SKILL_ASSESSMENT_SYSTEM = """You are a technical interviewer generating MCQ questions.
Return ONLY valid JSON. No markdown, no explanation, no code blocks."""

SKILL_ASSESSMENT_PROMPT = """Generate exactly {count} multiple choice questions about {skill} for a {level} level assessment.

Return ONLY this JSON structure:
{{
  "questions": [
    {{
      "question": "What does the 'let' keyword do in JavaScript?",
      "options": ["Declares a block-scoped variable", "Declares a function", "Imports a module", "Creates a class"],
      "correctAnswer": 0
    }}
  ]
}}

Rules:
- EXACTLY {count} questions specifically about {skill}
- Each question must have exactly 4 real, meaningful options (NOT placeholders like 'Option A' or 'A')
- correctAnswer is the 0-based index (0, 1, 2, or 3) of the correct option
- Questions must test real practical knowledge of {skill}
- Return ONLY the JSON object, nothing else"""


INTERVIEW_PREP_PROMPT = """Generate interview preparation for a role.

Role: {role}
Level: {level}
Skills: {skills}
Interview Type: {interview_type}  # technical|behavioral|hr|system_design

Return JSON with:
{{
  "questions": [
    {{"question": "...", "type": "technical|behavioral", "difficulty": "easy|medium|hard", "expected_answer": "..."}}
  ],
  "topics_to_review": ["topic1", "topic2"],
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
  "experience_bullets": [
    {{"company": "", "bullets": ["bullet1", "bullet2"]}}
  ],
  "skills_formatted": {{"technical": [], "soft": []}},
  "ats_keywords": ["keyword1", "keyword2"]
}}"""


class CareerBrain(Brain):
    """Handles career-related features: advice, roadmap, skill gap, interview prep, resume builder."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        intent = state.intent or "CAREER_ADVICE"
        context = state.context or {}
        query = state.query or ""

        if intent == "SKILL_ASSESSMENT":
            return await self._skill_assessment(state, context)
        elif intent == "INTERVIEW_PREP":
            return await self._interview_prep(state, context)
        elif intent == "RESUME_BUILDER":
            return await self._resume_builder(state, context)
        elif intent == "CAREER_ADVICE":
            # Structured context (from generateCareerRoadmap) → generate roadmap
            if context.get("current_role") and context.get("target_role"):
                return await self._career_advice(state, context)
            # Free-text chat → conversational reply
            return await self._chat_advice(state, query)
        else:
            return await self._chat_advice(state, query)

    async def _chat_advice(self, state: BrainState, query: str) -> BrainState:
        """Handle free-text career questions conversationally with user context."""
        clean_query = re.sub(r'^career advice:\s*', '', query, flags=re.IGNORECASE).strip()
        context = state.context or {}
        # Build user context string from injected profile data
        user_ctx_parts = []
        if context.get("user_name"): user_ctx_parts.append(f"Name: {context['user_name']}")
        if context.get("current_role"): user_ctx_parts.append(f"Current Role: {context['current_role']}")
        if context.get("target_role"): user_ctx_parts.append(f"Target Role: {context['target_role']}")
        if context.get("skills"): user_ctx_parts.append(f"Skills: {', '.join(context['skills'][:10])}")
        if context.get("ats_score"): user_ctx_parts.append(f"ATS Score: {context['ats_score']}%")
        if context.get("experience_years"): user_ctx_parts.append(f"Experience: {context['experience_years']}")
        if context.get("applications_count"): user_ctx_parts.append(f"Applications sent: {context['applications_count']}")
        if context.get("missing_skills"): user_ctx_parts.append(f"Missing skills: {', '.join(context['missing_skills'][:5])}")
        user_context = "\n".join(user_ctx_parts) if user_ctx_parts else "No profile data available."
        system = CAREER_CHAT_SYSTEM.format(user_context=user_context)
        # Inject RAG context if available
        rag_docs = context.get("rag_context", [])
        rag_text = "\n".join(d["text"] for d in rag_docs[:3]) if rag_docs else ""
        prompt = f"{clean_query}\n\nRelevant context:\n{rag_text}" if rag_text else clean_query
        try:
            reply = await ollama_service.generate(
                brain_name="career_advice",
                prompt=prompt,
                system=system,
                temperature=0.4,
                max_tokens=600,
            )
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            state.result = {"reply": reply, "intent": "CAREER_ADVICE", "rag_used": bool(rag_text)}
        except Exception:
            state.result = {"reply": "I'm having trouble right now. Please try again in a moment.", "intent": "CAREER_ADVICE"}
        return state

    async def _career_advice(self, state: BrainState, context: dict) -> BrainState:
        prompt = CAREER_ADVICE_PROMPT.format(
            current_role=context.get("current_role", "Software Engineer"),
            target_role=context.get("target_role", "Senior Software Engineer"),
            current_skills=", ".join(context.get("current_skills", ["Python", "JavaScript"])),
            experience_years=context.get("experience_years", 3),
            location=context.get("location", "Remote"),
        )
        try:
            result = await ollama_service.generate(
                brain_name="career_advice",
                prompt=prompt,
                system=CAREER_SYSTEM,
                temperature=0.3,
                max_tokens=2048,
            )
            state.result = self._parse_json(result)
        except Exception:
            state.result = self._fallback_career_advice(context)
            state.metadata["fallback"] = True
        return state

    async def _skill_assessment(self, state: BrainState, context: dict) -> BrainState:
        skill = context.get("skill", "Python")
        prompt = SKILL_ASSESSMENT_PROMPT.format(
            skill=skill,
            level=context.get("level", "intermediate"),
            count=context.get("count", 10),
        )
        try:
            result = await ollama_service.generate(
                brain_name="skill_assessment",
                prompt=prompt,
                system=SKILL_ASSESSMENT_SYSTEM,
                temperature=0.3,
                max_tokens=3000,
            )
            parsed = self._parse_json(result)
            questions = parsed.get("questions", []) if parsed else []
            # Validate: filter out placeholder options and wrong types
            valid = [
                q for q in questions
                if q.get("question") and
                isinstance(q.get("options"), list) and len(q["options"]) == 4 and
                isinstance(q.get("correctAnswer"), int) and 0 <= q["correctAnswer"] <= 3 and
                not any(o.strip() in ("A", "B", "C", "D") for o in q["options"])
            ]
            if len(valid) < 5:
                raise ValueError(f"Only {len(valid)} valid questions generated")
            state.result = {"questions": valid[:10]}
        except Exception as e:
            state.result = {"questions": [], "error": str(e)}
            state.metadata["fallback"] = True
        return state

    async def _interview_prep(self, state: BrainState, context: dict) -> BrainState:
        prompt = INTERVIEW_PREP_PROMPT.format(
            role=context.get("role", "Software Engineer"),
            level=context.get("level", "mid"),
            skills=", ".join(context.get("skills", ["Python", "JavaScript"])),
            interview_type=context.get("interview_type", "technical"),
        )
        try:
            result = await ollama_service.generate(
                brain_name="interview_prep",
                prompt=prompt,
                system=INTERVIEW_SYSTEM,
                temperature=0.3,
                max_tokens=2048,
            )
            state.result = self._parse_json(result)
            if not state.result:
                raise ValueError("Empty result")
        except Exception:
            state.result = {"questions": [], "topics_to_review": [], "tips": []}
            state.metadata["fallback"] = True
        return state

    async def _resume_builder(self, state: BrainState, context: dict) -> BrainState:
        prompt = RESUME_BUILDER_PROMPT.format(
            personal_info=json.dumps(context.get("personal_info", {})),
            experience=json.dumps(context.get("experience", [])),
            education=json.dumps(context.get("education", [])),
            skills=json.dumps(context.get("skills", {})),
            target_role=context.get("target_role", "Software Engineer"),
        )
        try:
            result = await ollama_service.generate(
                brain_name="resume_builder",
                prompt=prompt,
                system=RESUME_BUILDER_SYSTEM,
                temperature=0.3,
                max_tokens=2048,
            )
            state.result = self._parse_json(result)
        except Exception:
            state.result = {"summary": "", "experience_bullets": [], "skills_formatted": {}, "ats_keywords": []}
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

    def _fallback_career_advice(self, context: dict) -> dict:
        return {
            "career_path": [
                {"step": 1, "title": "Learn " + context.get("target_role", "target role"), "skills_to_learn": context.get("current_skills", []), "estimated_months": 12}
            ],
            "skill_gaps": ["Identify missing skills from job descriptions"],
            "recommended_courses": [],
            "certifications": [],
            "timeline_months": 12,
            "advice": "Focus on building projects that demonstrate target role skills.",
        }


career_brain = CareerBrain()
"""Career Brain - handles career advice, roadmap, skill gap, interview prep, resume builder."""
import re
import json
from typing import Optional
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

CAREER_SYSTEM = """You are an expert career advisor for tech professionals.
Return ONLY valid JSON as specified. No extra text, no markdown, no explanation."""

CAREER_CHAT_SYSTEM = """You are ZyncJobs AI Career Coach — a friendly, expert career advisor.
Give concise, actionable advice on career planning, resume writing, interview prep, skill gaps, and salary negotiation.
Keep responses clear and encouraging. Use bullet points. Max 3-4 short paragraphs."""

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


SKILL_ASSESSMENT_PROMPT = """Generate skill assessment questions.

Skill: {skill}
Level: {level}  # beginner|intermediate|advanced
Question Count: {count}

Return JSON with:
{{
  "questions": [
    {{
      "question": "What is...?",
      "type": "multiple_choice|code|short_answer",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "A",
      "explanation": "Why this is correct"
    }}
  ]
}}"""


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

        # Free-text chat query (from Career Coach page) — return conversational reply
        has_structured_context = any(k in context for k in ("current_role", "target_role", "skill", "role", "personal_info"))
        if intent == "CAREER_ADVICE" and not has_structured_context:
            return await self._chat_advice(state, query)

        if intent == "CAREER_ADVICE":
            return await self._career_advice(state, context)
        elif intent == "SKILL_ASSESSMENT":
            return await self._skill_assessment(state, context)
        elif intent == "INTERVIEW_PREP":
            return await self._interview_prep(state, context)
        elif intent == "RESUME_BUILDER":
            return await self._resume_builder(state, context)
        else:
            return await self._chat_advice(state, query)

    async def _chat_advice(self, state: BrainState, query: str) -> BrainState:
        """Handle free-text career questions conversationally."""
        # Strip the routing prefix added by frontend
        clean_query = re.sub(r'^career advice:\s*', '', query, flags=re.IGNORECASE).strip()
        try:
            reply = await ollama_service.generate(
                brain_name="career_advice",
                prompt=clean_query,
                system=CAREER_CHAT_SYSTEM,
                temperature=0.4,
                max_tokens=600,
            )
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            state.result = {"reply": reply, "intent": "CAREER_ADVICE"}
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
        prompt = SKILL_ASSESSMENT_PROMPT.format(
            skill=context.get("skill", "Python"),
            level=context.get("level", "intermediate"),
            count=context.get("count", 5),
        )
        try:
            result = await ollama_service.generate(
                brain_name="skill_assessment",
                prompt=prompt,
                system=CAREER_SYSTEM,
                temperature=0.3,
                max_tokens=2048,
            )
            state.result = self._parse_json(result)
            if not state.result:
                raise ValueError("Empty result")
        except Exception:
            state.result = {"questions": [], "error": "Assessment generation failed"}
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
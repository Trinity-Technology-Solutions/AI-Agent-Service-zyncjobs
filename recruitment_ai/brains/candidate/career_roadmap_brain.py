"""Career Roadmap Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict

ROADMAP_SYSTEM = """You are an expert career roadmap planner.
Return ONLY valid JSON as specified. No extra text, no markdown, no explanation."""

ROADMAP_PROMPT = """Generate a detailed career roadmap.

Current Role: {current_role}
Target Role: {target_role}
Current Skills: {current_skills}
Experience Years: {experience_years}
Location: {location}

Return JSON with:
{{
  "roadmap": [
    {{
      "phase": 1,
      "title": "Phase title",
      "duration_months": 3,
      "goals": ["goal1", "goal2"],
      "skills_to_learn": ["skill1", "skill2"],
      "milestones": ["milestone1"],
      "resources": [{{"name": "resource", "url": ""}}]
    }}
  ],
  "total_duration_months": 18,
  "certifications": [{{"name": "cert", "provider": "AWS|Google|Microsoft", "priority": "high|medium"}}],
  "salary_progression": [{{"phase": 1, "expected_range": "80k-100k USD"}}],
  "market_trends": ["trend1", "trend2"],
  "advice": "Personalized career advice"
}}"""


class CareerRoadmapBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        ctx = state.context_data
        prefs = ctx.user_preferences
        resume = ctx.resume

        prompt = ROADMAP_PROMPT.format(
            current_role=prefs.get("current_role", "Software Engineer"),
            target_role=prefs.get("target_role", state.request.query or "Senior Software Engineer"),
            current_skills=", ".join(resume.skills or prefs.get("current_skills", [])) or "Not specified",
            experience_years=prefs.get("experience_years", 0),
            location=prefs.get("location", "Remote"),
        )

        try:
            rag_docs = state.retrieved_documents.chunks or state.context.get("rag_context", [])
            rag_text = "\n".join(d.get("text", "") for d in rag_docs[:2]) if rag_docs else ""
            full_prompt = f"{prompt}\n\nAdditional context:\n{rag_text}" if rag_text else prompt
            result = await llm_service.generate(
                brain_name="career_advice",
                prompt=full_prompt,
                system=ROADMAP_SYSTEM,
                temperature=0.3,
                max_tokens=2048,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(
                response=parsed if parsed else self._fallback(prefs),
                metadata={"rag_used": bool(rag_text)},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response=self._fallback(prefs),
                metadata={"fallback_reason": str(e)},
            )

    def _fallback(self, prefs: dict) -> dict:
        target = prefs.get("target_role", "target role")
        return {
            "roadmap": [{
                "phase": 1,
                "title": f"Foundation for {target}",
                "duration_months": 6,
                "goals": ["Build core skills", "Complete relevant projects"],
                "skills_to_learn": [], "milestones": ["First project completed"], "resources": [],
            }],
            "total_duration_months": 12, "certifications": [],
            "salary_progression": [], "market_trends": [],
            "advice": f"Focus on building practical experience for {target}.",
        }


career_roadmap_brain = CareerRoadmapBrain()

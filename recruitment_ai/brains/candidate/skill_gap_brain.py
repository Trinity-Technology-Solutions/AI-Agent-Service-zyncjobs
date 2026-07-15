"""Skill Gap Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict

SKILL_GAP_SYSTEM = """You are a technical skill gap analyst.
Return ONLY valid JSON as specified. No extra text, no markdown, no explanation."""

SKILL_GAP_PROMPT = """Analyze the skill gap between a candidate's current skills and their target role.

Current Role: {current_role}
Target Role: {target_role}
Current Skills: {current_skills}
Experience Years: {experience_years}

Return JSON with:
{{
  "missing_skills": [
    {{"skill": "skill_name", "priority": "critical|important|nice_to_have", "reason": "why needed"}}
  ],
  "existing_relevant_skills": ["skill1", "skill2"],
  "gap_score": 0-100,
  "learning_resources": [
    {{"skill": "skill_name", "resource": "Course/Book/Project", "platform": "Coursera|Udemy|YouTube|GitHub", "estimated_weeks": 4}}
  ],
  "quick_wins": ["skill you can learn fast"],
  "summary": "Brief gap analysis summary"
}}"""


class SkillGapBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        ctx = state.context_data
        prefs = ctx.user_preferences
        resume = ctx.resume

        prompt = SKILL_GAP_PROMPT.format(
            current_role=prefs.get("current_role", "Software Engineer"),
            target_role=prefs.get("target_role", state.request.query or "Senior Software Engineer"),
            current_skills=", ".join(resume.skills or prefs.get("current_skills", [])) or "Not specified",
            experience_years=prefs.get("experience_years", 0),
        )

        try:
            rag_docs = state.retrieved_documents.chunks or state.context.get("rag_context", [])
            rag_text = "\n".join(d.get("text", "") for d in rag_docs[:2]) if rag_docs else ""
            full_prompt = f"{prompt}\n\nAdditional context:\n{rag_text}" if rag_text else prompt
            result = await llm_service.generate(
                brain_name="skill_assessment",
                prompt=full_prompt,
                system=SKILL_GAP_SYSTEM,
                temperature=0.2,
                max_tokens=2048,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(
                response=parsed if parsed else self._fallback(prefs, resume),
                metadata={"rag_used": bool(rag_text)},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response=self._fallback(prefs, resume),
                metadata={"fallback_reason": str(e)},
            )

    def _fallback(self, prefs: dict, resume) -> dict:
        target = prefs.get("target_role", "target role")
        return {
            "missing_skills": [{"skill": f"Core {target} skills", "priority": "critical", "reason": "Required for role"}],
            "existing_relevant_skills": resume.skills or prefs.get("current_skills", []),
            "gap_score": 50, "learning_resources": [], "quick_wins": [],
            "summary": f"Skill gap analysis for {target} is temporarily unavailable.",
        }


skill_gap_brain = SkillGapBrain()

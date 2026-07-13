"""Skill Gap Brain - dedicated brain for skill gap analysis (architecture doc section 11)."""
import re
import json
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

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
    """Dedicated skill gap analysis brain — Qwen3:8b + Skill Taxonomy."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        context = state.context or {}
        prompt = SKILL_GAP_PROMPT.format(
            current_role=context.get("current_role", "Software Engineer"),
            target_role=context.get("target_role", state.query or "Senior Software Engineer"),
            current_skills=", ".join(context.get("current_skills", [])) or "Not specified",
            experience_years=context.get("experience_years", 0),
        )
        try:
            rag_docs = context.get("rag_context", [])
            rag_text = "\n".join(d["text"] for d in rag_docs[:2]) if rag_docs else ""
            full_prompt = f"{prompt}\n\nAdditional context:\n{rag_text}" if rag_text else prompt
            result = await ollama_service.generate(
                brain_name="skill_assessment",  # qwen3:8b per architecture Brain-to-Model table
                prompt=full_prompt,
                system=SKILL_GAP_SYSTEM,
                temperature=0.2,
                max_tokens=2048,
            )
            parsed = self._parse_json(result)
            state.result = parsed if parsed else self._fallback(context)
            state.metadata["rag_used"] = bool(rag_text)
        except Exception as e:
            state.result = self._fallback(context)
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
        return {}

    def _fallback(self, context: dict) -> dict:
        target = context.get("target_role", "target role")
        current = context.get("current_skills", [])
        return {
            "missing_skills": [{"skill": f"Core {target} skills", "priority": "critical", "reason": "Required for role"}],
            "existing_relevant_skills": current,
            "gap_score": 50,
            "learning_resources": [],
            "quick_wins": [],
            "summary": f"Skill gap analysis for {target} is temporarily unavailable.",
        }


skill_gap_brain = SkillGapBrain()

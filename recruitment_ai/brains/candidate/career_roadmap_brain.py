"""Career Roadmap Brain - dedicated brain for career roadmap generation (architecture doc section 11)."""
import re
import json
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

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
    """Dedicated career roadmap brain — Llama3.1:8b + Skill Gap + Market Trends."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        context = state.context or {}
        prompt = ROADMAP_PROMPT.format(
            current_role=context.get("current_role", "Software Engineer"),
            target_role=context.get("target_role", state.query or "Senior Software Engineer"),
            current_skills=", ".join(context.get("current_skills", [])) or "Not specified",
            experience_years=context.get("experience_years", 0),
            location=context.get("location", "Remote"),
        )
        try:
            rag_docs = context.get("rag_context", [])
            rag_text = "\n".join(d["text"] for d in rag_docs[:2]) if rag_docs else ""
            full_prompt = f"{prompt}\n\nAdditional context:\n{rag_text}" if rag_text else prompt
            result = await ollama_service.generate(
                brain_name="career_advice",  # llama3.1:8b per architecture Brain-to-Model table
                prompt=full_prompt,
                system=ROADMAP_SYSTEM,
                temperature=0.3,
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
        return {
            "roadmap": [
                {
                    "phase": 1,
                    "title": f"Foundation for {target}",
                    "duration_months": 6,
                    "goals": ["Build core skills", "Complete relevant projects"],
                    "skills_to_learn": [],
                    "milestones": ["First project completed"],
                    "resources": [],
                }
            ],
            "total_duration_months": 12,
            "certifications": [],
            "salary_progression": [],
            "market_trends": [],
            "advice": f"Focus on building practical experience for {target}.",
        }


career_roadmap_brain = CareerRoadmapBrain()

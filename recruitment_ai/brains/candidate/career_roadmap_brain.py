"""Career Roadmap Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict

from recruitment_ai.prompts import get_system_prompt, get_prompt as _get_prompt

ROADMAP_SYSTEM = get_system_prompt("roadmap")


class CareerRoadmapBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        ctx = state.context_data
        prefs = ctx.user_preferences
        resume = ctx.resume

        prompt = _get_prompt("roadmap_prompt",
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
        current = prefs.get("current_role", "current role")
        skills = prefs.get("current_skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        skills_list = skills if isinstance(skills, list) else []
        skill_details = [
            {"skill": s, "why": f"Core skill needed for {target}", "resource": f"Industry standard for {target} roles", "resource_url": "https://roadmap.sh", "platform": "Docs"}
            for s in (skills_list[:3] if skills_list else [f"Core {target} fundamentals", "Industry tools", "Best practices"])
        ]
        return {
            "roadmap": [{
                "phase": 1,
                "title": f"Foundation for {target}",
                "duration_months": 6,
                "goals": [f"Build core skills for {target}", "Complete relevant projects", f"Understand {current} to {target} transition"],
                "skills_to_learn": [s["skill"] for s in skill_details] + [f"Advanced {target} concepts", "Portfolio projects", "Industry networking"],
                "skill_details": skill_details,
                "milestones": [f"Master core {target} concepts", f"Build 2 portfolio projects", f"Earn entry-level certification"],
                "certifications": [{"name": f"{target} Foundation Certification", "provider": "Industry Org", "priority": "high"}],
                "salary_range": "Entry level range",
            }],
            "total_duration_months": 12,
            "transferable_skills": [s for s in skills_list[:3]] if skills_list else [],
            "certifications": [{"name": f"{target} Foundation Certification", "provider": "Industry Org", "priority": "high"}],
            "salary_progression": [{"phase": 1, "expected_range": "Entry level range"}, {"phase": 2, "expected_range": "Mid level range"}],
            "market_trends": [f"High demand for {target} in 2025", "Remote-friendly opportunities available"],
            "market_demand": {"demand_level": "High", "job_openings": "10,000+", "remote_percentage": 60, "top_companies": ["Industry leaders"], "growth_rate": "20% YoY"},
            "advice": f"Focus on building practical experience for {target}. Your background in {current} gives you transferable skills that can accelerate this transition.",
        }


career_roadmap_brain = CareerRoadmapBrain()

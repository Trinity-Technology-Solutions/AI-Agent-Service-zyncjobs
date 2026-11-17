"""Skill Gap Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json

from recruitment_ai.prompts import get_system_prompt, get_prompt as _get_prompt

SKILL_GAP_SYSTEM = get_system_prompt("skill_gap")


class SkillGapBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        ctx = state.context_data
        prefs = ctx.user_preferences
        resume = ctx.resume
        raw_skills = resume.skills or prefs.get("current_skills", [])
        current_skills = [raw_skills] if isinstance(raw_skills, str) else raw_skills

        prompt = _get_prompt("skill_gap_prompt",
            current_role=prefs.get("current_role", "Not specified"),
            target_role=prefs.get("target_role", "Not specified"),
            current_skills=", ".join(skill for skill in current_skills if isinstance(skill, str)) or "Not specified",
            experience_years=prefs.get("experience_years", 0),
            experience_evidence=prefs.get("experience_evidence", "Not provided"),
            project_evidence=prefs.get("project_evidence", "Not provided"),
            education_evidence=prefs.get("education_evidence", "Not provided"),
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
            parsed = self._parse_skill_gap_result(result)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("missing_skills"), list):
                # Small local models occasionally ignore the JSON shape on their
                # first pass. Retry once with a short, explicit schema reminder.
                retry_result = await llm_service.generate(
                    brain_name="skill_assessment",
                    prompt=(
                        f"{full_prompt}\n\nYour previous response was not valid for this task. "
                        'Return ONLY this JSON object shape: {"missing_skills":[{"skill":"Docker",'
                        '"priority":"important","reason":"brief role-based reason"}]}. '
                        'If there are no supported gaps, return {"missing_skills":[]}. No markdown or extra text.'
                    ),
                    system=SKILL_GAP_SYSTEM,
                    temperature=0.0,
                    max_tokens=1024,
                )
                parsed = self._parse_skill_gap_result(retry_result)
            # Older frontend prompts requested a bare JSON array. Normalize that
            # legacy shape so a valid AI answer doesn't become an unavailable result.
            if not isinstance(parsed, dict) or not isinstance(parsed.get("missing_skills"), list):
                return BrainResult(
                    response=self._fallback(prefs, resume),
                    metadata={"analysis_available": False, "fallback_reason": "Invalid skill-gap response schema"},
                    execution_time=time.perf_counter() - start,
                )

            evidence_parts = [
                *(skill for skill in current_skills if isinstance(skill, str)),
                prefs.get("experience_evidence", ""),
                prefs.get("project_evidence", ""),
                prefs.get("education_evidence", ""),
            ]
            missing_skills = []
            for item in parsed["missing_skills"]:
                skill = item if isinstance(item, str) else item.get("skill") if isinstance(item, dict) else None
                if not isinstance(skill, str) or not skill.strip() or self._evidence_has_skill(skill, evidence_parts):
                    continue
                missing_skills.append(item)
            parsed["missing_skills"] = missing_skills[:8]
            parsed["existing_relevant_skills"] = parsed.get("existing_relevant_skills") or current_skills
            parsed["analysis_available"] = True
            return BrainResult(
                response=parsed,
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
        current_skills = resume.skills or prefs.get("current_skills", [])
        if isinstance(current_skills, str):
            current_skills = [current_skills]
        return {
            "missing_skills": [],
            "existing_relevant_skills": current_skills,
            "gap_score": None, "learning_resources": [], "quick_wins": [],
            "analysis_available": False,
            "summary": f"Skill gap analysis for {target} is temporarily unavailable.",
        }

    @staticmethod
    def _parse_skill_gap_result(raw: str):
        parsed = validate_json(raw)
        # Older frontend prompts requested a bare JSON array; normalize that
        # legacy shape so a valid AI answer doesn't become an unavailable result.
        if isinstance(parsed, list):
            return {"missing_skills": parsed}
        return parsed

    @staticmethod
    def _normalize_skill(skill: str) -> str:
        value = re.sub(r"[^a-z0-9]+", " ", skill.lower()).strip()
        aliases = {"rest": "rest api", "rest apis": "rest api", "restful api": "rest api", "restful apis": "rest api", "js": "javascript", "ts": "typescript", "ci cd": "cicd"}
        return aliases.get(value, value)

    @classmethod
    def _evidence_has_skill(cls, skill: str, evidence_parts: list[str]) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", skill.lower()).strip()
        canonical = cls._normalize_skill(skill)
        variants = {normalized, canonical}
        if canonical == "rest api":
            variants.update({"rest", "rest apis", "restful api", "restful apis"})
        if canonical == "cicd":
            variants.add("ci cd")

        for part in evidence_parts:
            if not isinstance(part, str):
                continue
            text = re.sub(r"[^a-z0-9]+", " ", part.lower()).strip()
            padded = f" {text} "
            if any(f" {variant} " in padded for variant in variants if variant):
                return True
        return False


skill_gap_brain = SkillGapBrain()

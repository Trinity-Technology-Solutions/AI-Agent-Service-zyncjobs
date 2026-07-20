"""Resume Parser Brain — full enterprise pipeline.
Pipeline: BrainState.request → LLM → JSON Validator → BrainResult
"""
import re
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict, ensure_json_fields
from recruitment_ai.prompts import get_prompt, get_system_prompt
from recruitment_ai.brains.candidate.skill_keywords import SKILL_KEYWORDS


class ResumeParserBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        from recruitment_ai.utils.ocr import extract_text
        raw_content = state.request.file_content or state.file_content or state.request.query or state.query or ""
        file_type = state.request.file_type or state.file_type or "txt"
        content = extract_text(raw_content, file_type) if file_type != "txt" else raw_content

        if not content.strip():
            return BrainResult(success=False, response={"error": "No resume content provided"})

        prompt = get_prompt("resume_parser_prompt", resume_text=content[:8000])
        system = get_system_prompt("resume_parser")

        try:
            raw = await llm_service.generate(
                brain_name="resume_parser",
                prompt=prompt,
                system=system,
                temperature=0.1,
                max_tokens=3000,
            )
            parsed = validate_json_strict(raw, "object") or {}
            data = self._validate(parsed, content)
            return BrainResult(
                response=data,
                metadata={"parser": "llm", "extracted": True},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response=self._fallback_parse(content),
                metadata={"parser": "fallback", "fallback_reason": str(e)},
                execution_time=time.perf_counter() - start,
            )

    def _validate(self, parsed: dict, content: str) -> dict:
        if not parsed.get("name") or len(parsed["name"].split()) < 2:
            parsed["name"] = self._extract_name(content) or "Unknown"
        if not parsed.get("email"):
            match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", content)
            parsed["email"] = match.group() if match else ""
        if not parsed.get("phone"):
            match = re.search(r"[\+\(]?[\d\-\(\)\s]{8,}", content)
            parsed["phone"] = match.group().strip() if match else ""
        return parsed

    def _extract_name(self, content: str) -> str:
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        for line in lines[:5]:
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$", line):
                return line
        return ""

    def _fallback_parse(self, content: str) -> dict:
        name = self._extract_name(content)
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", content)
        phone = re.search(r"[\+\(]?[\d\-\(\)\s]{8,}", content)
        skills = [kw for kw in SKILL_KEYWORDS if re.search(rf"\b{kw}\b", content, re.IGNORECASE)]
        return {
            "name": name or "Unknown",
            "email": email.group() if email else "",
            "phone": phone.group().strip() if phone else "",
            "location": "",
            "title": (content.split("\n")[0] if content else ""),
            "skills": skills, "softSkills": [], "tools": [],
            "workExperiences": [], "educations": [],
            "summary": content[:200],
        }


resume_parser_brain = ResumeParserBrain()

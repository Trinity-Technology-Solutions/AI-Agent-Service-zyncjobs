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

        if not self._is_resume(content):
            return BrainResult(success=False, response={"error": "The uploaded file does not appear to be a resume. Please upload a valid resume."})

        prompt = get_prompt("resume_parser_prompt", resume_text=content[:12000])
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

    def _is_resume(self, content: str) -> bool:
        """Return True only if content has enough resume signals to be worth parsing."""
        text = content.lower()
        # Must have at least one contact signal (email or phone)
        has_contact = bool(
            re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", content) or
            re.search(r"(?:\+\d{1,3}[\s\-]?)?\d[\d\s\-\.\(\)]{8,}\d", content)
        )
        # Must have at least 2 of these resume section keywords (prefix match)
        section_keywords = [
            "experience", "education", "skill", "summary", "objective",
            "profile", "certification", "project", "internship",
            "employment", "qualification", "achievement",
        ]
        section_hits = sum(1 for kw in section_keywords if kw in text)
        return has_contact and section_hits >= 2

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
            normalized = line
            if line == line.upper() and len(line) > 1:
                normalized = line.title()
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z.]*){1,3}$", normalized):
                return line
        return ""

    def _classify_education(self, line: str) -> dict:
        is_school = re.search(r"\b(hsc|sslc|10th|12th|higher secondary|secondary school)\b", line, re.IGNORECASE)
        year_match = re.search(r"(\d{4})", line)
        year = year_match.group(1) if year_match else ""
        if is_school:
            return {"type": "school", "school": line, "class": "", "year": year, "percentage": ""}
        return {"type": "college", "college": line, "degree": "", "year": year, "gpa": ""}

    def _fallback_parse(self, content: str) -> dict:
        name = self._extract_name(content)
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", content)
        # Phone: match international/Indian numbers, skip postal codes and date ranges
        phone_match = None
        for m in re.finditer(r"(?:\+\d{1,3}[\s\-]?)?\d[\d\s\-\.\(\)]{8,}\d", content):
            candidate = m.group().strip()
            digits = re.sub(r"\D", "", candidate)
            # Skip if it looks like a date range (e.g. 2020-2024) or postal code (<7 digits)
            if re.match(r"^(19|20)\d{2}.{0,3}(19|20)\d{2}$", candidate.strip()):
                continue
            if len(digits) < 7:
                continue
            phone_match = candidate
            break
        skills = [kw for kw in SKILL_KEYWORDS if re.search(rf"\b{kw}\b", content, re.IGNORECASE)]
        # Education: extract lines with degree/institution keywords, classify by type
        edu_lines = [
            l.strip() for l in content.split("\n")
            if re.search(r"\b(b\.?tech|b\.?e|m\.?tech|mba|bachelor|master|phd|diploma|university|college|institute|school|icam|loyola|iit|nit|hsc|sslc|10th|12th|higher secondary|secondary)\b", l, re.IGNORECASE)
        ]
        educations = [self._classify_education(l) for l in edu_lines[:3]]
        return {
            "name": name or "Unknown",
            "email": email.group() if email else "",
            "phone": phone_match or "",
            "location": "",
            "title": (content.split("\n")[0] if content else ""),
            "skills": skills, "softSkills": [], "tools": [],
            "workExperiences": [], "educations": educations,
            "projects": [], "certifications": [], "competitions": [],
            "summary": content[:200],
        }


resume_parser_brain = ResumeParserBrain()

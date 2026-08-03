"""Dedicated resume endpoints — parse, build, edit, score, upload, PDF export."""
import logging
import base64
from typing import Optional
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from recruitment_ai.schemas.brains import (
    ResumeParseRequest, ResumeParseResponse,
    ResumeBuildRequest,
    ResumeEditRequest,
)
from recruitment_ai.schemas.api import ExecuteResponse
from recruitment_ai.schemas.error import ErrorResponse
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.workflows.recruitment_graph import graph
from recruitment_ai.brains.base import BrainState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/resume", tags=["resume"])

# Optional auth — allows unauthenticated requests (token fetch may fail on frontend)
_optional_bearer = HTTPBearer(auto_error=False)

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_optional_bearer),
) -> dict:
    if not credentials:
        return {"user_id": "anonymous", "role": "candidate", "email": None}
    try:
        from jose import jwt as _jwt
        from recruitment_ai.config.settings import settings
        payload = _jwt.decode(
            credentials.credentials,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return {
            "user_id": payload.get("sub"),
            "role": payload.get("role", "candidate"),
            "email": payload.get("email"),
        }
    except Exception:
        return {"user_id": "anonymous", "role": "candidate", "email": None}


def _build_state(request_data: dict, user: dict, intent: str) -> dict:
    return {
        "user": {"id": user.get("user_id"), "email": user.get("email"), "role": user.get("role", "candidate")},
        "session": {"id": None},
        "conversation": {},
        "context_data": {"user_preferences": {}},
        "retrieved_documents": {},
        "provider_info": {},
        "execution": {},
        "request": request_data,
        "response": None,
        "intent": intent,
        "error": None,
        "metadata": {},
        "query": request_data.get("query", ""),
        "user_id": user.get("user_id"),
        "user_role": user.get("role", "candidate"),
    }


@router.post("/parse", response_model=ResumeParseResponse)
async def parse_resume(request: ResumeParseRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state(
        {"file_content": request.file_content, "file_type": request.file_type, "query": "Parse my resume"},
        user, "RESUME_PARSER",
    ))
    r = result.get("result") or {}
    parsed = r.get("parsed_data") or r
    # Fallback: if graph returned empty, use regex on file content
    if not parsed or not any(parsed.get(k) for k in ("name", "email", "workExperiences", "skills")):
        try:
            from recruitment_ai.utils.ocr import extract_text
            text = extract_text(request.file_content or "", request.file_type or "txt")
            if text.strip():
                parsed = _regex_fallback(text)
        except Exception as fe:
            logger.warning("Regex fallback in /parse failed: %s", fe)
    return ResumeParseResponse(success=result.get("error") is None, parsed=parsed, error=result.get("error"))


@router.post("/build", response_model=dict)
async def build_resume(request: ResumeBuildRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state(
        {"sections": request.sections, "query": "Build a resume"},
        user, "RESUME_BUILDER",
    ))
    return {"success": result.get("error") is None, "result": result.get("result"), "error": result.get("error")}


@router.post("/edit", response_model=dict)
async def edit_resume(request: ResumeEditRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state(
        {"section": request.section, "instruction": request.instruction,
         "current_content": request.current_content, "query": "Edit my resume"},
        user, "RESUME_EDIT",
    ))
    return {"success": result.get("error") is None, "result": result.get("result"), "error": result.get("error")}


@router.post("/upload")
async def upload_resume(file: UploadFile = File(...), user: dict = Depends(get_optional_user)):
    """Upload resume file to S3 and return the URL."""
    try:
        from recruitment_ai.services.s3 import s3_service
        data = await file.read()
        url = await s3_service.upload(data, file.filename or "resume.pdf", folder="resumes")
        if url:
            return {"success": True, "url": url, "filename": file.filename}
        return {"success": False, "error": "S3 upload failed"}
    except Exception as e:
        logger.warning("Resume upload failed: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/parse-upload", response_model=ResumeParseResponse)
async def parse_upload_resume(file: UploadFile = File(...), user: dict = Depends(get_optional_user)):
    """Upload and immediately parse a resume file. Always re-parses — no cache."""
    data = b""
    try:
        import json
        from recruitment_ai.brains.candidate.resume_parser_brain import resume_parser_brain
        from recruitment_ai.brains.shared.brain_state import BrainState as BS, RequestInfo

        data = await file.read()
        filename = file.filename or "resume.pdf"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "pdf"
        file_content_b64 = base64.b64encode(data).decode("utf-8")

        state = BS(
            user={"id": user.get("user_id"), "email": user.get("email"), "role": user.get("role", "candidate")},
            request=RequestInfo(query="Parse my resume", file_content=file_content_b64, file_type=ext),
            query="Parse my resume",
            file_content=file_content_b64,
            file_type=ext,
            user_id=user.get("user_id"),
            user_role=user.get("role", "candidate"),
            intent="RESUME_PARSER",
        )
        result = await resume_parser_brain.run(state)
        parsed = result.response or {}
        # Always also run regex fallback and merge — fills gaps the brain misses
        try:
            from recruitment_ai.utils.ocr import extract_text
            text = extract_text(file_content_b64, ext)
            if text.strip():
                regex_parsed = _regex_fallback(text)
                # Merge: regex fills empty fields, brain takes priority
                for key in ("name", "email", "phone", "location", "linkedin", "summary"):
                    if not parsed.get(key) and regex_parsed.get(key):
                        parsed[key] = regex_parsed[key]
                for key in ("skills", "softSkills", "tools", "workExperiences", "educations", "projects", "certifications"):
                    if not parsed.get(key) and regex_parsed.get(key):
                        parsed[key] = regex_parsed[key]
                    elif parsed.get(key) and regex_parsed.get(key):
                        # Merge lists — add items from regex that are missing from brain
                        existing = parsed[key] or []
                        existing_set = {json.dumps(e, sort_keys=True) if isinstance(e, dict) else e for e in existing}
                        for item in (regex_parsed.get(key) or []):
                            item_key = json.dumps(item, sort_keys=True) if isinstance(item, dict) else item
                            if item_key not in existing_set:
                                existing.append(item)
                        parsed[key] = existing
        except Exception as fe:
            logger.warning("Regex merge failed: %s", fe)
        return ResumeParseResponse(success=True, parsed=parsed, error=result.error)
    except Exception as e:
        logger.warning("parse-upload failed: %s", e)
        try:
            if data:
                from recruitment_ai.utils.ocr import extract_text
                file_content_b64 = base64.b64encode(data).decode("utf-8")
                ext = (file.filename or "resume.pdf").rsplit(".", 1)[-1].lower()
                text = extract_text(file_content_b64, ext)
                if text.strip():
                    return ResumeParseResponse(success=True, parsed=_regex_fallback(text), error=None)
        except Exception as fe:
            logger.warning("Regex fallback also failed: %s", fe)
        return ResumeParseResponse(success=False, parsed={}, error=str(e))


def _extract_section(text: str, name: str, after: int = 0) -> tuple[str, int]:
    """Find section content by locating header, returning (content, end_pos)."""
    import re
    # Find section header: e.g. "Experience" or "Work Experience" at start of line
    pat = rf"(?:^|\n)\s*(?:{name})\s*[:]?\s*\n"
    m = re.search(pat, text[after:], re.IGNORECASE)
    if not m:
        return ("", after)
    start = after + m.end()
    # Find next section header
    next_pat = r"(?:^|\n)\s*"
    next_pat += r"(?:Skills?|Experience|Education|Work\s+(?:History|Experience)?|Employment"
    next_pat += r"|Projects?|Certifications?|Summary|Profile|Objective|About Me"
    next_pat += r"|Career Objective|Technical Skills|Languages?|Achievements?|Awards?)\s*[:]?\s*\n"
    n = re.search(next_pat, text[start:], re.IGNORECASE)
    if n:
        content = text[start:start + n.start()]
    else:
        content = text[start:]
    return (content.strip(), start)


def _regex_fallback(text: str) -> dict:
    """Pure regex resume parser — works without LLM."""
    import re

    pos = 0
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Name: first line that looks like a proper name (not email/phone/url)
    name = ""
    for line in lines[:6]:
        normalized = line.title() if line.isupper() else line
        if (re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z.]+){1,3}$", normalized)
                and len(normalized) < 50
                and "@" not in normalized
                and not re.search(r"\d{5,}", normalized)):
            name = normalized
            break

    # Contact fields
    email_m = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    phone_m = re.search(r"[\+]?[\d][\d\s\-().]{7,15}[\d]", text)
    linkedin_m = re.search(r"linkedin\.com/in/[\w-]+", text, re.IGNORECASE)

    # Location: labeled OR "City, Country/State" pattern in first 10 lines
    location = ""
    loc_labeled = re.search(r"(?:Location|Address|City)[:\s]+([^\n,]{2,40})", text, re.IGNORECASE)
    if loc_labeled:
        location = loc_labeled.group(1).strip()
    else:
        for line in lines[:10]:
            if (re.match(r"^[A-Za-z\s]+,\s*[A-Za-z\s]+$", line)
                    and len(line) < 40 and "@" not in line
                    and not re.search(r"\d", line)):
                location = line
                break

    # Summary
    summary = ""
    summary_content, pos = _extract_section(
        text, r"(?:Summary|Profile|Objective|About Me|Career Objective)", pos
    )
    if summary_content:
        summary = " ".join(summary_content.split())[:500]

    # Skills
    skills_content, pos = _extract_section(
        text, r"(?:Technical\s+)?Skills?(?:\s+&\s+\w+)?", pos
    )
    skills: list = []
    if skills_content:
        skills = [
            s.strip().strip(",•·|-–")
            for s in re.split(r"[,\n•·|/]", skills_content) if 2 < len(s.strip()) < 40
        ]
        skills = [s for s in skills if s and not re.match(r"^(and|or|the|with|using|etc)$", s, re.I)][:30]

    # Work Experience
    exp_content, pos = _extract_section(
        text, r"(?:Work\s+)?(?:Experience|Employment|Work\s+History)", pos
    )
    work_experiences: list = []
    if exp_content:
        # Split into job blocks on blank lines
        raw_blocks = [b.strip() for b in re.split(r"\n\s*\n", exp_content) if b.strip()]
        job_blocks = []
        for block in raw_blocks:
            blines = [l.strip() for l in block.split("\n") if l.strip()]
            if not blines:
                continue
            # If first line looks like a date, prepend to previous block
            if re.search(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}", blines[0], re.IGNORECASE):
                if job_blocks:
                    job_blocks[-1] += "\n" + block
                else:
                    job_blocks.append(block)
            else:
                job_blocks.append(block)

        for block in job_blocks[:8]:
            blines = [l.strip() for l in block.split("\n") if l.strip()]
            if not blines:
                continue
            title_line = blines[0]
            company = ""
            title = title_line
            # Split "Title at Company" / "Title | Company" / "Title - Company"
            split_m = re.match(r"^(.+?)\s+(?:at|@|\|)\s+(.+)$", title_line, re.IGNORECASE)
            if not split_m:
                split_m = re.match(r"^(.+?)\s+[–\-]\s+(.+)$", title_line)
            if split_m:
                title = split_m.group(1).strip()
                company = split_m.group(2).strip()
            date_str = ""
            remaining = blines[1:]
            for i, l in enumerate(remaining):
                date_m = re.search(
                    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|\d{4})"
                    r"\s*[-–to]+\s*"
                    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{4}|Present|Current|Now)",
                    l, re.IGNORECASE
                )
                if date_m:
                    date_str = date_m.group(0)
                    remaining = [x for j, x in enumerate(remaining) if j != i]
                    break
            if not company and remaining:
                first_rem = remaining[0]
                if (not first_rem.startswith(("•", "-", "–", "*"))
                        and len(first_rem) < 60
                        and not re.search(r"\d{4}", first_rem)
                        and not first_rem.startswith(("http", "www"))):
                    company = first_rem
                    remaining = remaining[1:]
            bullets = [l.lstrip("•·-–* ") for l in remaining if len(l.lstrip("•·-–* ")) > 8][:6]
            if title and len(title) > 2 and " " in title:
                work_experiences.append({
                    "jobTitle": title,
                    "company": company,
                    "date": date_str,
                    "descriptions": bullets,
                })

    # Education
    edu_content, pos = _extract_section(
        text, r"(?:Education|Educational Qualification)", pos
    )
    educations: list = []
    if edu_content:
        raw_edu_blocks = [b.strip() for b in re.split(r"\n\s*\n", edu_content) if b.strip()]
        edu_blocks = []
        for block in raw_edu_blocks:
            block = block.strip()
            if not block:
                continue
            blines = [l.strip() for l in block.split("\n") if l.strip()]
            if not blines:
                continue
            first_line = blines[0]
            if re.search(r"^(CGPA|GPA|Percentage|Grade|Score|\d{4})", first_line, re.IGNORECASE):
                if edu_blocks:
                    edu_blocks[-1] += "\n" + block
                else:
                    edu_blocks.append(block)
            else:
                edu_blocks.append(block)

        degree_keywords = [
            r"\bB\.?\s*Tech", r"\bM\.?\s*Tech", r"\bB\.?\s*E\.?", r"\bM\.?\s*E\.?",
            r"\bB\.?\s*Sc", r"\bM\.?\s*Sc",
            r"\bMBA\b", r"\bBCA\b", r"\bMCA\b", r"\bB\.?\s*Com", r"\bM\.?\s*Com",
            r"\bBachelors?", r"\bMasters?", r"\bPhD\b", r"\bPh\.\s*D",
            r"\bDiploma", r"\bHSC\b", r"\bSSC\b", r"\bClass\s*X{1,2}", r"\b10th\b", r"\b12th\b",
            r"\bBachelor\s+of", r"\bMaster\s+of",
        ]
        degree_pattern = "|".join(degree_keywords)

        for block in edu_blocks[:5]:
            blines = [l.strip() for l in block.split("\n") if l.strip()]
            if not blines:
                continue
            year = ""
            year_m = re.search(r"(20\d{2}|19\d{2})", block)
            if year_m:
                year = year_m.group(0)
            grade = ""
            grade_m = re.search(r"(?:CGPA|GPA|Percentage|Grade|Score)[:\s]*([\d.]+\s*%?)", block, re.IGNORECASE)
            if grade_m:
                grade = grade_m.group(1).strip()

            degree = ""
            school = ""
            degree_line_idx = -1

            for idx, l in enumerate(blines):
                if re.search(degree_pattern, l, re.IGNORECASE):
                    degree = l.strip()
                    degree_line_idx = idx
                    break
            if not degree:
                for idx, l in enumerate(blines):
                    if not re.search(r"^(\d{4}|CGPA|GPA|Percentage|Grade|Score)", l, re.IGNORECASE):
                        degree = l.strip()
                        degree_line_idx = idx
                        break
            # If no degree keyword was matched in the fallback, treat it as school
            if degree and not re.search(degree_pattern, degree, re.IGNORECASE):
                school = degree
                degree = ""

            if degree_line_idx >= 0:
                for idx in range(degree_line_idx + 1, len(blines)):
                    l = blines[idx]
                    skip = re.search(
                        degree_pattern + r"|^(\d{4})|^(CGPA|GPA|Percentage|Grade|Score)", l, re.IGNORECASE
                    )
                    if not skip and len(l) > 3 and len(l) < 80 and not re.match(r"^[\d.\s]+%?$", l):
                        school = l.strip()
                        break
            if not school:
                for l in blines:
                    skip = re.search(
                        degree_pattern + r"|^(\d{4})|^(CGPA|GPA|Percentage|Grade|Score)", l, re.IGNORECASE
                    )
                    if not skip and len(l) > 3 and len(l) < 80 and not re.match(r"^[\d.\s]+%?$", l) and l != degree:
                        school = l.strip()
                        break

            if degree or school:
                educations.append({
                    "degree": degree if degree else school,
                    "school": school,
                    "date": year,
                    "grade": grade,
                })

    # Projects
    proj_content, pos = _extract_section(text, r"Projects?", pos)
    projects: list = []
    if proj_content:
        proj_lines = [l.strip() for l in proj_content.split("\n") if l.strip()]
        for line in proj_lines[:5]:
            if len(line) > 10:
                proj_m = re.match(r"^(.+?)\s+(?:[–\-—])\s+(.+)$", line)
                if proj_m:
                    projects.append({"name": proj_m.group(1).strip(), "description": proj_m.group(2).strip()})
                else:
                    projects.append({"name": line, "description": ""})

    # Certifications
    cert_content, pos = _extract_section(text, r"Certifications?", pos)
    certifications: list = []
    if cert_content:
        certs = [c.strip() for c in re.split(r"[,;\n]", cert_content) if len(c.strip()) >= 3]
        certifications = [{"name": c} for c in certs[:10]]

    return {
        "name": name,
        "email": email_m.group(0) if email_m else "",
        "phone": phone_m.group(0).strip() if phone_m else "",
        "location": location,
        "linkedin": linkedin_m.group(0) if linkedin_m else "",
        "title": "",
        "summary": summary,
        "skills": skills,
        "softSkills": [],
        "tools": [],
        "workExperiences": work_experiences,
        "educations": educations,
        "projects": projects,
        "certifications": certifications,
    }


@router.post("/score", response_model=dict)
async def score_resume(resume_text: str, job_description: str, user: dict = Depends(get_current_user)):
    """Score a resume against a job description."""
    result = await graph.ainvoke(_build_state(
        {"resume_text": resume_text, "job_description": job_description, "query": "Score my resume"},
        user, "ATS_SCORE",
    ))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "score": r.get("ats_score") or r.get("score"),
        "feedback": r.get("suggestions") or r.get("feedback"),
        "error": result.get("error"),
    }


@router.post("/export-pdf")
async def export_resume_pdf(data: dict, user: dict = Depends(get_current_user)):
    """Export resume data as a downloadable PDF."""
    try:
        from recruitment_ai.services.pdf_generator import generate_resume_pdf
        pdf_bytes = await generate_resume_pdf(data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=resume_{user.get('user_id', 'export')}.pdf"},
        )
    except Exception as e:
        logger.warning("PDF export failed: %s", e)
        return {"success": False, "error": str(e)}


@router.post("/cover-letter-pdf")
async def export_cover_letter_pdf(data: dict, user: dict = Depends(get_current_user)):
    """Export cover letter data as a downloadable PDF."""
    try:
        from recruitment_ai.services.pdf_generator import generate_cover_letter_pdf
        pdf_bytes = await generate_cover_letter_pdf(data)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=cover_letter_{user.get('user_id', 'export')}.pdf"},
        )
    except Exception as e:
        logger.warning("Cover letter PDF export failed: %s", e)
        return {"success": False, "error": str(e)}

"""Dedicated resume endpoints — parse, build, edit, score, upload, PDF export."""
import logging
from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import Response
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
    return ResumeParseResponse(success=result.get("error") is None, parsed=r.get("parsed_data") or r, error=result.get("error"))


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
async def upload_resume(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
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

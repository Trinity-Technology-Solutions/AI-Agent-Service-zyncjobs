"""Dedicated ATS endpoint — analyze resumes against job descriptions."""
import logging
from fastapi import APIRouter, Depends
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.workflows.recruitment_graph import graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/ats", tags=["ats"])


@router.post("/analyze", response_model=dict)
async def ats_analyze(resume_text: str, job_description: str, user: dict = Depends(get_current_user)):
    """Analyze a resume against a job description and return ATS score."""
    state = {
        "user": {"id": user.get("user_id"), "email": user.get("email"), "role": user.get("role", "candidate")},
        "query": f"ATS score check for resume",
        "user_id": user.get("user_id"),
        "user_role": user.get("role", "candidate"),
        "request": {"query": "ATS score check", "resume_text": resume_text, "job_description": job_description},
        "intent": "ATS_SCORE",
        "session": {"id": None}, "conversation": {}, "context_data": {}, "retrieved_documents": {},
        "provider_info": {}, "execution": {}, "response": None, "error": None, "metadata": {},
    }
    result = await graph.ainvoke(state)
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "ats_score": r.get("ats_score") or r.get("score") or 0,
        "keyword_match": r.get("keyword_match", {}),
        "formatting_score": r.get("formatting_score", 0),
        "section_completeness": r.get("section_completeness", 0),
        "experience_relevance": r.get("experience_relevance", 0),
        "suggestions": r.get("suggestions") or [],
        "passes_ats": r.get("passes_ats", False),
        "error": result.get("error"),
    }

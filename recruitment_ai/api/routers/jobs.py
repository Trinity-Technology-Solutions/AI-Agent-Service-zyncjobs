"""Dedicated job endpoints — JD generate, parse, match, ATS scan, recommend, search."""
import logging
from fastapi import APIRouter, Depends, Query
from typing import Optional
from recruitment_ai.schemas.brains import (
    JdGenerateRequest, JdGenerateResponse,
    JobMatchRequest, JobMatchResponse,
    AtsScanRequest, AtsScanResponse,
)
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.workflows.recruitment_graph import graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/jobs", tags=["jobs"])


def _build_state(request: dict, user: dict, intent: str) -> dict:
    return {
        "user": {"id": user.get("user_id"), "email": user.get("email"), "role": user.get("role", "candidate")},
        "session": {"id": None},
        "conversation": {},
        "context_data": {"user_preferences": {}},
        "retrieved_documents": {},
        "provider_info": {},
        "execution": {},
        "request": request,
        "response": None,
        "intent": intent,
        "error": None,
        "metadata": {},
        "query": request.get("query", ""),
        "user_id": user.get("user_id"),
        "user_role": user.get("role", "candidate"),
    }


@router.post("/generate", response_model=JdGenerateResponse)
async def generate_jd(request: JdGenerateRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "title": request.title, "company": request.company,
        "skills": request.skills, "experience_level": request.experience_level,
        "location": request.location, "employment_type": request.employment_type,
        "query": f"Generate a job description for {request.title}",
    }, user, "JD_GENERATOR"))
    r = result.get("result") or {}
    return JdGenerateResponse(success=result.get("error") is None, description=r.get("description") or r.get("reply"), error=result.get("error"))


@router.post("/match", response_model=JobMatchResponse)
async def match_job(request: JobMatchRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "resume_text": request.resume_text, "job_description": request.job_description,
        "query": "Find me a job match",
    }, user, "JOB_MATCH"))
    r = result.get("result") or {}
    return JobMatchResponse(
        success=result.get("error") is None,
        score=r.get("score") or r.get("match_score"),
        matched_skills=r.get("matched_skills") or [],
        missing_skills=r.get("missing_skills") or [],
        feedback=r.get("feedback") or r.get("reply"),
        error=result.get("error"),
    )


@router.post("/ats-scan", response_model=AtsScanResponse)
async def ats_scan(request: AtsScanRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "resume_text": request.resume_text, "job_description": request.job_description,
        "query": "What is my ATS score",
    }, user, "ATS_SCORE"))
    r = result.get("result") or {}
    return AtsScanResponse(
        success=result.get("error") is None,
        score=r.get("ats_score") or r.get("score"),
        keyword_matches=r.get("keyword_matches") or r.get("matched_keywords") or [],
        missing_keywords=r.get("missing_keywords") or [],
        suggestions=r.get("suggestions") or [],
        error=result.get("error"),
    )


@router.post("/recommend", response_model=dict)
async def recommend_jobs(skills: list[str] = [], experience: Optional[str] = None,
                          user: dict = Depends(get_current_user)):
    """Get AI-powered job recommendations based on skills and experience."""
    result = await graph.ainvoke(_build_state({
        "skills": skills, "experience": experience or "",
        "query": f"Recommend jobs for skills: {', '.join(skills)}",
    }, user, "JOB_MATCH"))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "recommendations": r.get("recommendations") or r.get("matched_jobs") or [],
        "error": result.get("error"),
    }


@router.post("/search", response_model=dict)
async def search_jobs(query: str = Query(..., description="Natural language job search query"),
                       location: Optional[str] = None,
                       user: dict = Depends(get_current_user)):
    """AI-powered job search using natural language."""
    search_query = f"{query} {location or ''}".strip()
    result = await graph.ainvoke(_build_state({
        "query": search_query, "search_location": location,
    }, user, "JOB_MATCH"))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "results": r.get("jobs") or r.get("recommendations") or [],
        "query": query,
        "error": result.get("error"),
    }


@router.post("/parse", response_model=dict)
async def parse_job_description(job_text: str, user: dict = Depends(get_current_user)):
    """Parse a job description text into structured fields."""
    result = await graph.ainvoke(_build_state({
        "job_text": job_text, "query": "Parse this job description",
    }, user, "JOB_PARSER"))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "parsed": r.get("parsed") or r,
        "error": result.get("error"),
    }

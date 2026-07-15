"""Dedicated recruiter endpoints — search, shortlist, interview questions, candidate ranking."""
from fastapi import APIRouter, Depends
from recruitment_ai.schemas.brains import (
    RecruiterSearchRequest, RecruiterSearchResponse,
)
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.workflows.recruitment_graph import graph

router = APIRouter(prefix="/ai/recruiter", tags=["recruiter"])


def _build_state(request: dict, user: dict, intent: str) -> dict:
    return {
        "user": {"id": user.get("user_id"), "email": user.get("email"), "role": "employer"},
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
        "user_role": "employer",
    }


@router.post("/search", response_model=RecruiterSearchResponse)
async def search_candidates(request: RecruiterSearchRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "criteria": request.criteria, "filters": request.filters or {},
        "query": request.criteria,
    }, user, "RECRUITER_SEARCH"))
    r = result.get("result") or {}
    return RecruiterSearchResponse(
        success=result.get("error") is None,
        candidates=r.get("candidates") or [],
        total_count=r.get("total_count", 0),
        query=request.criteria,
        error=result.get("error"),
    )




@router.post("/candidates/rank", response_model=dict)
async def rank_candidates(job_description: str, candidates: list = [], user: dict = Depends(get_current_user)):
    """Rank candidates by fit score for a given job."""
    result = await graph.ainvoke(_build_state({
        "job_description": job_description, "candidates": candidates,
        "query": "Rank candidates for job",
    }, user, "RECRUITER"))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "ranked_candidates": r.get("candidates") or r.get("ranked") or [],
        "error": result.get("error"),
    }

@router.post("/shortlist", response_model=RecruiterSearchResponse)
async def shortlist_candidates(request: RecruiterSearchRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "criteria": request.criteria, "filters": request.filters or {},
        "query": f"Shortlist candidates: {request.criteria}",
    }, user, "RECRUITER_SHORTLIST"))
    r = result.get("result") or {}
    return RecruiterSearchResponse(
        success=result.get("error") is None,
        candidates=r.get("candidates") or [],
        total_count=r.get("total_count", 0),
        query=request.criteria,
        error=result.get("error"),
    )

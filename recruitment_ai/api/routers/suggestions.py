"""AI suggestion endpoints — skill, title, and location suggestions."""
import logging
from fastapi import APIRouter, Depends
from typing import Optional
from pydantic import BaseModel
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.workflows.recruitment_graph import graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/suggest", tags=["suggestions"])


class SuggestRequest(BaseModel):
    context: str
    limit: int = 10


def _build_state(request: dict, user: dict) -> dict:
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
        "intent": "CHAT",
        "error": None,
        "metadata": {},
        "query": request.get("query", ""),
        "user_id": user.get("user_id"),
        "user_role": user.get("role", "candidate"),
    }


@router.post("/skills")
async def suggest_skills(request: SuggestRequest, user: dict = Depends(get_current_user)):
    """Suggest relevant skills based on job title or role."""
    result = await graph.ainvoke(_build_state({
        "query": f"Suggest skills for: {request.context}",
        "context": request.context, "type": "skills",
    }, user))
    r = result.get("result") or {}
    suggestions = r.get("suggestions") or r.get("reply", "").split("\n")
    return {"success": True, "suggestions": [s.strip() for s in suggestions if s.strip()][:request.limit]}


@router.post("/titles")
async def suggest_titles(request: SuggestRequest, user: dict = Depends(get_current_user)):
    """Suggest job titles based on skills or industry."""
    result = await graph.ainvoke(_build_state({
        "query": f"Suggest job titles for: {request.context}",
        "context": request.context, "type": "titles",
    }, user))
    r = result.get("result") or {}
    suggestions = r.get("suggestions") or r.get("reply", "").split("\n")
    return {"success": True, "suggestions": [s.strip() for s in suggestions if s.strip()][:request.limit]}


@router.post("/locations")
async def suggest_locations(request: SuggestRequest, user: dict = Depends(get_current_user)):
    """Suggest locations for a given role or industry."""
    result = await graph.ainvoke(_build_state({
        "query": f"Suggest locations for: {request.context}",
        "context": request.context, "type": "locations",
    }, user))
    r = result.get("result") or {}
    suggestions = r.get("suggestions") or r.get("reply", "").split("\n")
    return {"success": True, "suggestions": [s.strip() for s in suggestions if s.strip()][:request.limit]}

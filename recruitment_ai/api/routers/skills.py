"""Dedicated skill endpoints — assessment, gap analysis, career roadmap."""
from fastapi import APIRouter, Depends
from recruitment_ai.schemas.brains import (
    SkillAssessmentRequest, SkillAssessmentResponse,
    SkillGapRequest, SkillGapResponse,
    CareerRoadmapRequest, CareerRoadmapResponse,
)
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.workflows.recruitment_graph import graph

router = APIRouter(prefix="/ai/skills", tags=["skills"])


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


@router.post("/assessment", response_model=SkillAssessmentResponse)
async def skill_assessment(request: SkillAssessmentRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "skill": request.skill, "level": request.level,
        "num_questions": request.num_questions,
        "query": f"Assess my {request.skill} skills",
    }, user, "SKILL_ASSESSMENT"))
    r = result.get("result") or {}
    return SkillAssessmentResponse(
        success=result.get("error") is None,
        questions=r.get("questions") or [],
        skill=request.skill,
        error=result.get("error"),
    )


@router.post("/gap-analysis", response_model=SkillGapResponse)
async def skill_gap(request: SkillGapRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "target_role": request.target_role, "current_skills": request.current_skills,
        "query": f"Analyze skill gaps for {request.target_role}",
    }, user, "SKILL_GAP"))
    r = result.get("result") or {}
    return SkillGapResponse(
        success=result.get("error") is None,
        current_skills=r.get("current_skills") or request.current_skills,
        required_skills=r.get("required_skills") or [],
        gaps=r.get("gaps") or r.get("missing_skills") or [],
        recommendations=r.get("recommendations") or [],
        error=result.get("error"),
    )


@router.post("/career-roadmap", response_model=CareerRoadmapResponse)
async def career_roadmap(request: CareerRoadmapRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "target_role": request.target_role, "current_role": request.current_role,
        "time_horizon": request.time_horizon,
        "query": f"Create a career roadmap for {request.target_role}",
    }, user, "CAREER_ROADMAP"))
    r = result.get("result") or {}
    return CareerRoadmapResponse(
        success=result.get("error") is None,
        roadmap=r.get("roadmap") or [],
        milestones=r.get("milestones") or [],
        error=result.get("error"),
    )

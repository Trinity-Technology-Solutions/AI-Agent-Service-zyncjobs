"""Dedicated chat endpoints — general chat, career advice, interview prep, cover letter, recruiter chat."""
import logging
from fastapi import APIRouter, Depends, Query
from typing import Optional
from recruitment_ai.schemas.brains import (
    ChatRequest, ChatResponse,
    InterviewPrepRequest, InterviewPrepResponse,
    CoverLetterRequest, CoverLetterResponse,
)
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.workflows.recruitment_graph import graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai/chat", tags=["chat"])


def _build_state(request: dict, user: dict, intent: str) -> dict:
    return {
        "user": {"id": user.get("user_id"), "email": user.get("email"), "role": user.get("role", "candidate")},
        "session": {"id": request.get("session_id")},
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
        "query": request.get("query", request.get("message", "")),
        "user_id": user.get("user_id"),
        "user_role": user.get("role", "candidate"),
        "session_id": request.get("session_id"),
    }


@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "query": request.message, "message": request.message,
        "conversation_id": request.conversation_id,
        "session_id": request.session_id,
    }, user, "CHAT"))
    r = result.get("result") or {}
    return ChatResponse(
        success=result.get("error") is None,
        reply=r.get("reply") or r.get("response") or r.get("text"),
        conversation_id=result.get("conversation_id") or request.conversation_id,
        error=result.get("error"),
    )


@router.post("/interview-prep", response_model=InterviewPrepResponse)
async def interview_prep(request: InterviewPrepRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "job_title": request.job_title, "company": request.company,
        "interview_type": request.interview_type, "skills": request.skills,
        "query": f"Prepare me for a {request.job_title} interview",
    }, user, "INTERVIEW_PREP"))
    r = result.get("result") or {}
    return InterviewPrepResponse(
        success=result.get("error") is None,
        questions=r.get("questions") or [],
        tips=r.get("tips") or [],
        error=result.get("error"),
    )


@router.post("/recruiter-chat", response_model=ChatResponse)
async def recruiter_chat(message: str, session_id: Optional[str] = None,
                          user: dict = Depends(get_current_user)):
    """Recruiter-specific chat with candidate search context."""
    result = await graph.ainvoke(_build_state({
        "message": message, "query": f"recruiter: {message}",
        "session_id": session_id,
    }, {"user_id": user.get("user_id"), "email": user.get("email"), "role": "employer"}, "RECRUITER"))
    r = result.get("result") or {}
    return ChatResponse(
        success=result.get("error") is None,
        reply=r.get("reply") or r.get("search_strategy") or r.get("response"),
        conversation_id=session_id,
        error=result.get("error"),
    )


@router.post("/suggest", response_model=dict)
async def suggest_content(context: str, content_type: str = "skills",
                           user: dict = Depends(get_current_user)):
    """AI-powered suggestions for skills, titles, or content."""
    result = await graph.ainvoke(_build_state({
        "context": context, "content_type": content_type,
        "query": f"Suggest {content_type} for: {context}",
    }, user, "CHAT"))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "suggestions": r.get("suggestions") or r.get("reply", "").split("\n"),
        "error": result.get("error"),
    }




@router.post("/interview-questions", response_model=dict)
async def interview_questions(job_title: str, skills: str = "", interview_type: str = "technical", user: dict = Depends(get_current_user)):
    """Generate interview questions for a given role."""
    result = await graph.ainvoke(_build_state({
        "job_title": job_title, "skills": skills.split(",") if skills else [],
        "interview_type": interview_type,
        "query": f"Prepare interview questions for {job_title}",
    }, user, "INTERVIEW_PREP"))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "questions": r.get("questions") or [],
        "tips": r.get("tips") or [],
        "error": result.get("error"),
    }

@router.post("/cover-letter", response_model=CoverLetterResponse)
async def cover_letter(request: CoverLetterRequest, user: dict = Depends(get_current_user)):
    result = await graph.ainvoke(_build_state({
        "job_title": request.job_title, "company": request.company,
        "resume_text": request.resume_text, "tone": request.tone,
        "query": f"Write a cover letter for {request.job_title} at {request.company}",
    }, user, "COVER_LETTER"))
    r = result.get("result") or {}
    return CoverLetterResponse(
        success=result.get("error") is None,
        letter=r.get("letter") or r.get("reply") or r.get("response"),
        error=result.get("error"),
    )

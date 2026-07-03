from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from app.services import resume_service

router = APIRouter()


class ImproveRequest(BaseModel):
    resume_text: str
    job_description: str = ""


class ImproveResponse(BaseModel):
    improved_resume: str
    ats_score: int | None
    summary: str
    skills_suggested: list[str]
    grammar_issues: list[dict]


@router.post("/improve", response_model=ImproveResponse)
async def improve_resume(request: ImproveRequest):
    return await resume_service.improve_resume(
        resume_text=request.resume_text,
        job_description=request.job_description,
    )


class ParseRequest(BaseModel):
    resume_text: str


@router.post("/parse")
def parse_resume(request: ParseRequest):
    return resume_service.parse_resume(resume_text=request.resume_text)


class ATSScoreRequest(BaseModel):
    resume_text: str
    job_description: str


@router.post("/ats-score")
def ats_score(request: ATSScoreRequest):
    return resume_service.ats_score(
        resume_text=request.resume_text,
        job_description=request.job_description,
    )


class LayoutBlock(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


class HybridParseRequest(BaseModel):
    resume_text: str
    layout_blocks: Optional[list[LayoutBlock]] = None


class HybridParseResponse(BaseModel):
    profile: dict
    summary: str
    skills: dict
    experience: list
    education: list
    projects: list
    certifications: list
    achievements: list
    languages: list


@router.post("/hybrid-parse", response_model=HybridParseResponse)
def hybrid_parse_resume(request: HybridParseRequest):
    blocks = [b.model_dump() for b in request.layout_blocks] if request.layout_blocks else None
    return resume_service.hybrid_parse_resume(resume_text=request.resume_text, layout_blocks=blocks)


# ─── Hybrid ATS Score v2 ───────────────────────────────────────────

class ATSV2Response(BaseModel):
    score: float
    rule_score: float
    ai_score: float
    components: dict
    matching_keywords: list[str]
    missing_keywords: list[str]
    suggestions: list[str]
    grammar_issues: list[dict]
    keyword_optimization: list[str]
    reason: str


@router.post("/ats-score-v2", response_model=ATSV2Response)
async def ats_score_v2(request: ATSScoreRequest):
    return await resume_service.ats_score_v2(
        resume_text=request.resume_text,
        job_description=request.job_description,
    )


# ─── Resume Intelligence ──────────────────────────────────────────

class IntelligenceRequest(BaseModel):
    resume_json: dict


@router.post("/intelligence")
def analyze_intelligence(request: IntelligenceRequest):
    return resume_service.analyze_intelligence(
        resume_json=request.resume_json,
    )


# ─── Grammar Checker (Pure AI) ────────────────────────────────────

class GrammarRequest(BaseModel):
    text: str
    mode: str = "check"


@router.post("/grammar")
def grammar_check(request: GrammarRequest):
    return resume_service.grammar_check(
        text=request.text,
        mode=request.mode,
    )
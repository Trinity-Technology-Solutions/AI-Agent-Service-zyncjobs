from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from app.ranking.candidate_ranking_service import (
    calculate_rule_score,
    calculate_ai_score,
    merge_scores,
    rank_candidates,
)

router = APIRouter()


class CandidateData(BaseModel):
    skills: list[str] = []
    workExperiences: list[dict] = []
    educations: list[dict] = []
    projects: list[dict] = []
    certifications: list[dict] = []
    location: str = ""
    noticePeriod: str = ""
    expectedSalary: str = ""
    profile: dict = {}
    featuredSkills: list[dict] = []


class JobData(BaseModel):
    skills: list[str] = []
    experience: str = ""
    education: str = ""
    location: str = ""
    salary: str = ""
    title: str = ""


class RuleScoreRequest(BaseModel):
    candidate: CandidateData
    job: JobData


class AIScoreRequest(BaseModel):
    candidate: CandidateData
    job: JobData


class HybridScoreRequest(BaseModel):
    candidate: CandidateData
    job: JobData


class RankRequest(BaseModel):
    candidates: list[CandidateData]
    job: JobData


@router.post("/rule-score")
def rule_score(request: RuleScoreRequest):
    c = request.candidate.model_dump()
    j = request.job.model_dump()
    return calculate_rule_score(c, j)


@router.post("/ai-score")
def ai_score(request: AIScoreRequest):
    c = request.candidate.model_dump()
    j = request.job.model_dump()
    return calculate_ai_score(c, j)


@router.post("/hybrid-score")
def hybrid_score(request: HybridScoreRequest):
    c = request.candidate.model_dump()
    j = request.job.model_dump()
    rule = calculate_rule_score(c, j)
    ai = calculate_ai_score(c, j)
    return merge_scores(rule, ai)


@router.post("/rank")
def rank(request: RankRequest):
    clist = [c.model_dump() for c in request.candidates]
    j = request.job.model_dump()
    return {"rankings": rank_candidates(clist, j)}

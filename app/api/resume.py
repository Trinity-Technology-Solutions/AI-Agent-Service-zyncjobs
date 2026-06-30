from fastapi import APIRouter
from pydantic import BaseModel
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

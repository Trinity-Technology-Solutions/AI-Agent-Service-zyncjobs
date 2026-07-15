"""Async task endpoints — submit background Celery jobs and poll results."""
from fastapi import APIRouter, Depends
from celery.result import AsyncResult
from pydantic import BaseModel
from typing import Optional
from recruitment_ai.auth.jwt_handler import get_current_user
from recruitment_ai.services.celery_app import celery_app
from recruitment_ai.schemas.error import ErrorResponse

router = APIRouter(prefix="/ai/tasks", tags=["tasks"])


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str = "pending"


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, user: dict = Depends(get_current_user)):
    result = AsyncResult(task_id, app=celery_app)
    resp = TaskStatusResponse(task_id=task_id, status=result.status)
    if result.ready():
        try:
            resp.result = result.get()
        except Exception as e:
            resp.error = str(e)
            resp.status = "FAILURE"
    return resp


@router.post("/resume/parse", response_model=TaskSubmitResponse)
async def submit_resume_parse(file_content: str, file_type: str = "pdf",
                               user: dict = Depends(get_current_user)):
    from recruitment_ai.services.tasks import process_resume
    task = process_resume.delay(file_content, file_type, user_id=user.get("user_id"))
    return TaskSubmitResponse(task_id=task.id)


@router.post("/ats/batch", response_model=TaskSubmitResponse)
async def submit_batch_ats(resumes: list[dict], job_description: str,
                            user: dict = Depends(get_current_user)):
    from recruitment_ai.services.tasks import batch_ats_score
    task = batch_ats_score.delay(resumes, job_description)
    return TaskSubmitResponse(task_id=task.id)


@router.post("/jobs/generate", response_model=TaskSubmitResponse)
async def submit_jd_generate(title: str, company: Optional[str] = None,
                              skills: Optional[list[str]] = None,
                              experience_level: Optional[str] = None,
                              location: Optional[str] = None,
                              user: dict = Depends(get_current_user)):
    from recruitment_ai.services.tasks import generate_jd
    task = generate_jd.delay(title, company, skills, experience_level, location)
    return TaskSubmitResponse(task_id=task.id)

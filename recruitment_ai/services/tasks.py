"""Celery tasks — background processing for long-running AI operations.
Each task wraps a brain execution and stores result in Redis (Celery backend).
"""
import asyncio
import logging
from typing import Optional
from celery import Task
from recruitment_ai.services.celery_app import celery_app

logger = logging.getLogger(__name__)


class AsyncBrainTask(Task):
    """Base task that sets up the async event loop for brain execution."""

    _loop = None

    def run_async(self, coro):
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)


@celery_app.task(bind=True, base=AsyncBrainTask, name="process_resume")
def process_resume(self, file_content: str, file_type: str, user_id: Optional[str] = None):
    """Parse a resume in the background and return structured data."""
    import time
    from recruitment_ai.brains.base import BrainState
    from recruitment_ai.brains.candidate.resume_parser_brain import resume_parser_brain

    start = time.perf_counter()
    state = BrainState(
        query="Parse my resume",
        file_content=file_content,
        file_type=file_type,
        user_id=user_id,
    )
    state.request.query = "Parse my resume"
    state.request.file_content = file_content
    state.request.file_type = file_type

    try:
        result = self.run_async(resume_parser_brain.run(state))
        elapsed = time.perf_counter() - start
        logger.info("Resume parsed in %.2fs (user=%s)", elapsed, user_id)
        return {
            "success": result.success,
            "data": result.response,
            "execution_time": elapsed,
            "tokens": result.tokens,
        }
    except Exception as e:
        logger.error("Resume parse failed: %s", e)
        return {"success": False, "error": str(e)}


@celery_app.task(bind=True, base=AsyncBrainTask, name="batch_ats_score")
def batch_ats_score(self, resumes: list[dict], job_description: str):
    """Score multiple resumes against a job description in batch."""
    import time
    from recruitment_ai.brains.base import BrainState
    from recruitment_ai.brains.candidate.ats_brain import ats_brain

    results = []
    for i, resume in enumerate(resumes):
        try:
            state = BrainState(
                query="What is my ATS score",
                file_content=resume.get("text", ""),
                context={"resume": resume, "job_description": job_description},
            )
            state.request.query = "What is my ATS score"
            result = self.run_async(ats_brain.run(state))
            results.append({
                "index": i,
                "success": result.success,
                "score": result.response.get("ats_score") if result.response else None,
                "error": result.error,
            })
        except Exception as e:
            results.append({"index": i, "success": False, "error": str(e)})

    logger.info("Batch ATS scored %d/%d resumes", sum(1 for r in results if r["success"]), len(resumes))
    return {"results": results, "total": len(resumes)}


@celery_app.task(bind=True, base=AsyncBrainTask, name="generate_jd")
def generate_jd(self, title: str, company: Optional[str] = None, skills: Optional[list] = None,
                experience_level: Optional[str] = None, location: Optional[str] = None):
    """Generate a job description in the background."""
    import time
    from recruitment_ai.brains.base import BrainState
    from recruitment_ai.brains.employer.jd_generator_brain import jd_generator_brain

    state = BrainState(
        query=f"Generate a job description for {title}",
        context={"title": title, "company": company, "skills": skills or [],
                 "experience_level": experience_level, "location": location},
    )
    state.request.query = state.query

    start = time.perf_counter()
    try:
        result = self.run_async(jd_generator_brain.run(state))
        elapsed = time.perf_counter() - start
        return {"success": result.success, "description": result.response, "execution_time": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e)}


@celery_app.task(bind=True, base=AsyncBrainTask, name="process_job_match")
def process_job_match(self, resume_text: str, job_description: str):
    """Match a resume to a job description in the background."""
    import time
    from recruitment_ai.brains.base import BrainState
    from recruitment_ai.brains.candidate.job_matching_brain import job_matching_brain

    state = BrainState(query="Find me a job match")
    state.request.query = "Find me a job match"

    start = time.perf_counter()
    try:
        result = self.run_async(job_matching_brain.run(state))
        elapsed = time.perf_counter() - start
        return {"success": result.success, "match": result.response, "execution_time": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e)}

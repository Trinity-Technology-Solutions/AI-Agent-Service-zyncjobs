from recruitment_ai.api.routers.resume import router as resume_router
from recruitment_ai.api.routers.jobs import router as jobs_router
from recruitment_ai.api.routers.chat import router as chat_router
from recruitment_ai.api.routers.recruiter import router as recruiter_router
from recruitment_ai.api.routers.skills import router as skills_router

__all__ = ["resume_router", "jobs_router", "chat_router", "recruiter_router", "skills_router"]

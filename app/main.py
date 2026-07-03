from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config.settings import settings
from app.api.chat import router as chat_router
from app.api.chatbot_v2 import router as chatbot_v2_router
from app.api.resume import router as resume_router
from app.api.career import router as career_router
from app.api.recruiter import router as recruiter_router
from app.api.interview import router as interview_router
from app.api.job import router as job_router
from app.api.knowledge import router as knowledge_router
from app.api.ranking import router as ranking_router
from app.api.monitoring import router as monitoring_router
from app.api.controllers.dashboard import router as dashboard_router
from app.api.controllers.system import router as system_router
from app.gateway.service_registry import service_registry
from app.services import resume_service, career_service, interview_service, recruiter_service, job_service, chat_service
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.error_handler import register_error_handlers
from app.knowledge.knowledge_base import knowledge_base
from app.memory.memory_manager import memory
from app.memory.cache import prompt_cache
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    for svc in [
        ("resume_service", resume_service, "Resume AI Services"),
        ("career_service", career_service, "Career AI Services"),
        ("interview_service", interview_service, "Interview AI Services"),
        ("recruiter_service", recruiter_service, "Recruiter AI Services"),
        ("job_service", job_service, "Job Match AI Services"),
        ("chat_service", chat_service, "Conversational AI Services"),
    ]:
        service_registry.register(name=svc[0], service=svc[1], version="v1", description=svc[2])

    BANNER = f"""
{'='*50}
  {settings.APP_NAME}
{'='*50}
  Version      : {settings.APP_VERSION}
  LLM          : Ollama ({settings.OLLAMA_MODEL})
  Services     : {len(service_registry.list())} registered
  Knowledge    : {knowledge_base.document_count} documents
  Memory       : Loaded
  Cache        : {prompt_cache.size} entries
  Gateway      : Ready
{'='*50}
"""
    logger.info(BANNER)
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

register_error_handlers(app)
app.add_middleware(LoggingMiddleware)

# Controllers
app.include_router(dashboard_router, prefix="")
app.include_router(system_router, prefix="")

# API routes
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(chatbot_v2_router, prefix="/api/v1/chatbot", tags=["Chatbot V2"])
app.include_router(resume_router, prefix="/api/v1/resume", tags=["Resume"])
app.include_router(career_router, prefix="/api/v1/career", tags=["Career"])
app.include_router(recruiter_router, prefix="/api/v1/recruiter", tags=["Recruiter"])
app.include_router(interview_router, prefix="/api/v1/interview", tags=["Interview"])
app.include_router(job_router, prefix="/api/v1/job", tags=["Job"])
app.include_router(knowledge_router, prefix="/api/v1/knowledge", tags=["Knowledge"])
app.include_router(ranking_router, prefix="/api/v1/ranking", tags=["Ranking"])
app.include_router(monitoring_router, prefix="/api/v1/admin/ai", tags=["AI Monitoring"])

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.api.chatbot_v2 import router as chatbot_v2_router
from app.api.knowledge import router as knowledge_router
from app.api.monitoring import router as monitoring_router
from app.api.controllers.dashboard import router as dashboard_router
from app.api.controllers.system import router as system_router
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.error_handler import register_error_handlers
from app.knowledge.knowledge_base import knowledge_base
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} | LLM: {settings.OLLAMA_MODEL} | Knowledge: {knowledge_base.document_count} docs")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_error_handlers(app)
app.add_middleware(LoggingMiddleware)

# Controllers
app.include_router(dashboard_router, prefix="")
app.include_router(system_router, prefix="")

# API routes
app.include_router(chatbot_v2_router, prefix="/api/v1/chatbot", tags=["Chatbot V2"])
app.include_router(knowledge_router, prefix="/api/v1/knowledge", tags=["Knowledge"])
app.include_router(monitoring_router, prefix="/api/v1/admin/ai", tags=["AI Monitoring"])

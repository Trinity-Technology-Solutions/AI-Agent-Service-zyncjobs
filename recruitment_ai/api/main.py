"""Main FastAPI application for recruitment AI platform."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from recruitment_ai.config.settings import settings
from recruitment_ai.auth.jwt_handler import get_current_user, create_access_token
from recruitment_ai.shared.master_brain import master_brain
from recruitment_ai.shared.brain import BrainState
from recruitment_ai.vector.store import vector_store
from recruitment_ai.api.middleware import error_handler_middleware, request_logger_middleware
from pydantic import BaseModel
from typing import Optional


class ExecuteRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_role: str = "candidate"
    context: Optional[dict] = None
    file_content: Optional[str] = None
    file_type: Optional[str] = None


class ExecuteResponse(BaseModel):
    success: bool
    intent: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    metadata: dict = {}


class TokenRequest(BaseModel):
    user_id: str
    role: str = "candidate"
    email: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from recruitment_ai.database.connection import init_db
        await init_db()
    except Exception as e:
        print(f"Database init skipped (file-based storage): {e}")
    yield
    from recruitment_ai.shared.ollama_service import ollama_service
    await ollama_service.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(request_logger_middleware)
app.middleware("http")(error_handler_middleware)


@app.post("/auth/token")
async def create_token(request: TokenRequest):
    token = create_access_token({"sub": request.user_id, "role": request.role, "email": request.email})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/ai/execute", response_model=ExecuteResponse)
async def execute_ai(request: ExecuteRequest, user: dict = Depends(get_current_user)):
    state = BrainState(
        query=request.query,
        session_id=request.session_id,
        user_id=user.get("user_id"),
        user_role=request.user_role,
        context=request.context,
        file_content=request.file_content,
        file_type=request.file_type,
    )

    result_state = await master_brain.execute(state)

    return ExecuteResponse(
        success=result_state.error is None,
        intent=result_state.intent,
        result=result_state.result,
        error=result_state.error,
        metadata=result_state.metadata,
    )


@app.get("/health")
async def health():
    from recruitment_ai.shared.ollama_service import ollama_service
    ollama_ok = await ollama_service.health_check()
    return {
        "status": "healthy" if ollama_ok else "degraded",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "ollama": "connected" if ollama_ok else "unreachable",
    }


@app.get("/knowledge/stats")
async def knowledge_stats():
    return {
        "total_chunks": vector_store.count,
        "collection": settings.QDRANT_COLLECTION,
        "backend": "chromadb",
    }


@app.get("/metrics")
async def metrics():
    try:
        from prometheus_client import generate_latest, REGISTRY
        return Response(content=generate_latest(REGISTRY).decode("utf-8"), media_type="text/plain")
    except ImportError:
        return {"message": "Prometheus client not installed", "chunks": vector_store.count, "brains": list(master_brain.brains.keys())}


@app.get("/version")
async def version():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "brains": list(master_brain.brains.keys()),
        "knowledge_chunks": vector_store.count,
        "ollama_model": settings.OLLAMA_MODELS.get("chatbot", "unknown"),
    }


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
"""Main FastAPI application for recruitment AI platform."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from recruitment_ai.config.settings import settings
from recruitment_ai.auth.jwt_handler import get_current_user, create_access_token
from recruitment_ai.brains.master.master_brain import master_brain
from recruitment_ai.brains.base import BrainState
from recruitment_ai.workflows.recruitment_graph import graph
from recruitment_ai.vector.store import vector_store
from recruitment_ai.api.middleware import error_handler_middleware, request_logger_middleware, RateLimitMiddleware
from recruitment_ai.api.routers import (
    resume_router, jobs_router, chat_router, recruiter_router, skills_router,
)
from recruitment_ai.api.routers.tasks import router as tasks_router
from recruitment_ai.api.routers.admin import router as admin_router
from recruitment_ai.api.routers.knowledge import router as knowledge_router
from recruitment_ai.api.routers.ats import router as ats_router
from recruitment_ai.api.routers.suggestions import router as suggestions_router
from recruitment_ai.schemas import ExecuteRequest, ExecuteResponse, AuthRequest

# ── Prometheus metrics ───────────────────────────────────────────────
AI_REQUESTS = Counter(
    "zyncjobs_ai_requests_total",
    "Total AI execute requests",
    ["intent", "status"],
)
AI_LATENCY = Histogram(
    "zyncjobs_ai_latency_seconds",
    "AI execute request latency",
    ["intent"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
CACHE_HITS = Counter(
    "zyncjobs_cache_hits_total",
    "Redis cache hits",
    ["intent"],
)


# OAuth2 scheme — architecture doc: OAuth2 + JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/oauth2/token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.configure_logging()
    try:
        from recruitment_ai.database.connection import init_db
        await init_db()
    except Exception as e:
        print(f"Database init skipped (file-based storage): {e}")
    from recruitment_ai.services.cache_service import cache_service
    await cache_service.connect()
    await vector_store.connect()
    # OpenTelemetry instrumentation
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    except Exception as e:
        print(f"OpenTelemetry instrumentation skipped: {e}")
    yield
    await cache_service.close()
    await vector_store.close()
    from recruitment_ai.llm import llm_service
    await llm_service.close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(request_logger_middleware)
app.middleware("http")(error_handler_middleware)
app.add_middleware(RateLimitMiddleware)

# ── Domain routers ───────────────────────────────────────────────────
app.include_router(resume_router)
app.include_router(jobs_router)
app.include_router(chat_router)
app.include_router(recruiter_router)
app.include_router(skills_router)
app.include_router(tasks_router)
app.include_router(admin_router)
app.include_router(ats_router)
app.include_router(knowledge_router)
app.include_router(suggestions_router)


@app.post("/auth/token")
async def create_token(request: AuthRequest):
    token = create_access_token({"sub": request.user_id, "role": request.role, "email": request.email})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/auth/oauth2/token")
async def oauth2_token(form: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password flow — architecture doc: OAuth2 + JWT auth."""
    token = create_access_token({"sub": form.username, "role": "candidate", "email": form.username})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/ai/execute", response_model=ExecuteResponse)
async def execute_ai(request: ExecuteRequest, user: dict = Depends(get_current_user)):
    import time
    workflow_input = {
        # ── Enterprise sub-objects (provide all so LangGraph channels are never empty) ──
        "user": {"id": user.get("user_id"), "email": user.get("email"), "role": request.user_role},
        "session": {"id": request.session_id},
        "conversation": {},
        "context_data": {"user_preferences": request.context or {}},
        "retrieved_documents": {},
        "provider_info": {},
        "execution": {},
        "request": {"query": request.query, "file_content": request.file_content, "file_type": request.file_type},
        "response": None,
        "intent": None,
        "error": None,
        "metadata": {},
        # ── Backward-compat flat fields ──
        "query": request.query,
        "user_id": user.get("user_id"),
        "session_id": request.session_id,
        "conversation_id": request.session_id,
        "user_role": request.user_role,
        "context": request.context or {},
        "file_content": request.file_content,
        "file_type": request.file_type,
        "retrieved_context": [],
        "memory": [],
        "result": None,
        "provider": "",
        "model": "",
    }

    start = time.perf_counter()
    result = await graph.ainvoke(workflow_input)
    elapsed = time.perf_counter() - start

    intent = result.get("intent") or "UNKNOWN"
    status = "error" if result.get("error") else "success"
    AI_REQUESTS.labels(intent=intent, status=status).inc()
    AI_LATENCY.labels(intent=intent).observe(elapsed)
    if result.get("metadata", {}).get("cache_hit"):
        CACHE_HITS.labels(intent=intent).inc()

    return ExecuteResponse(
        success=result.get("error") is None,
        intent=result.get("intent"),
        result=result.get("result"),
        error=result.get("error"),
        metadata=result.get("metadata", {}),
    )


@app.get("/health")
async def health():
    from recruitment_ai.llm import llm_service
    provider_ok = await llm_service.health_check()
    return {
        "status": "healthy" if provider_ok else "degraded",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "provider": settings.LLM_PROVIDER,
        "model": settings.BEDROCK_MODEL if settings.LLM_PROVIDER == "bedrock" else settings._DEFAULT_MODEL,
        "llm": "connected" if provider_ok else "unreachable",
    }


@app.get("/knowledge/stats")
async def knowledge_stats():
    return {
        "total_chunks": vector_store.count,
        "collection": settings.QDRANT_COLLECTION,
        "backend": vector_store.backend,
        "qdrant_url": settings.QDRANT_URL,
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(REGISTRY).decode("utf-8"), media_type="text/plain")


@app.get("/version")
async def version():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "brains": list(master_brain.brains.keys()),
        "knowledge_chunks": vector_store.count,
        "llm_provider": settings.LLM_PROVIDER,
    }


@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
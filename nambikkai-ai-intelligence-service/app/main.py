import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, analyze
from app.api.routes.suggestions import router as suggestions_router
from app.api.routes.reports import router as reports_router
from app.api.routes.intelligence import router as intelligence_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

# Psycopg async mode requires SelectorEventLoop.
# On Windows, Python 3.8+ defaults to ProactorEventLoop which is incompatible.
# This must be set before any event loop is created.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # pyright: ignore[reportDeprecated]

configure_logging()
logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.data_sources.postgres import close_pool, open_pool
    try:
        await open_pool()
        from app.services.bulk_scanner import evaluate_platform_readiness
        asyncio.create_task(evaluate_platform_readiness("youtube"))
        asyncio.create_task(evaluate_platform_readiness("instagram"))
    except Exception as exc:
        logger.warning("PostgreSQL pool not opened at startup: %s", exc)

    # Load XGBoost model artifact if configured.
    # This is a one-time synchronous operation at startup — never done during requests.
    try:
        from app.ml.xgboost_service import load_model_if_configured
        load_model_if_configured()
    except Exception as exc:
        logger.warning("XGBoost model load failed at startup (non-fatal): %s", exc)

    yield
    await close_pool()


app = FastAPI(
    title="Nambikkai AI Intelligence Service",
    version="0.1.0",
    description="AI intelligence layer for the Nambikkai analytics dashboard.",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
# Parse comma-separated origins from configuration.
# The AI service sits behind the dashboard backend proxy, so only the
# backend's origin (and localhost for development) needs to be allowed.
# Wildcard "*" is intentionally NOT used when credentials may be involved.
_raw_origins = settings.CORS_ALLOWED_ORIGINS.strip()
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()] if _raw_origins else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "x-api-key", "x-user-role", "x-user-email"],
)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(suggestions_router)
app.include_router(reports_router)
app.include_router(intelligence_router)

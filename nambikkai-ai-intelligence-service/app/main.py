import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health
from app.api.routes.suggestions import router as suggestions_router
from app.api.routes.reports import router as reports_router
from app.api.routes.intelligence import router as intelligence_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

# Psycopg async mode requires SelectorEventLoop on Windows.
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
        logger.info("PostgreSQL connection pool initialized.")
    except Exception as exc:
        logger.warning("PostgreSQL pool not opened at startup: %s", exc)

    yield
    await close_pool()


app = FastAPI(
    title="Nambikkai AI Intelligence Service",
    version="0.2.0",
    description="AI Performance Insights and Intelligence service for the Nambikkai dashboard.",
    lifespan=lifespan,
)

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
app.include_router(suggestions_router)
app.include_router(reports_router)
app.include_router(intelligence_router)

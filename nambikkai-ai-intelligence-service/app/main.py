import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

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

get_settings()


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
    yield
    await close_pool()


app = FastAPI(
    title="Nambikkai AI Intelligence Service",
    version="0.1.0",
    description="AI intelligence layer for the Nambikkai analytics dashboard.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(suggestions_router)
app.include_router(reports_router)
app.include_router(intelligence_router)

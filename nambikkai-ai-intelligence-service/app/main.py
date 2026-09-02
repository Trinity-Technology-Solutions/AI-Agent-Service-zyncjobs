from fastapi import FastAPI
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.api.routes import health, analysis

configure_logging()

# Warm the settings cache at startup so the first request does not pay
# the cost of .env parsing. The result is intentionally discarded here;
# all consumers call get_settings() directly.
get_settings()

app = FastAPI(
    title="Nambikkai AI Intelligence Service",
    version="0.1.0",
    description="AI intelligence layer for the Nambikkai analytics dashboard.",
)

app.include_router(health.router)
app.include_router(analysis.router)

"""
Read-only PostgreSQL data source for Nambikkai AI Intelligence Service.

Reads exclusively from AI-specific tables (*_history_ai, *_content_ai).
Must NOT read from or modify dashboard source tables.

Responsibilities:
  - Open / close a connection pool (lifecycle tied to FastAPI lifespan).
  - Fetch history rows from *_history_ai tables.
  - Fetch content metadata from *_content_ai tables.
  - Join and return NormalizedContentRecord objects.

This module must NOT:
  - Call an LLM.
  - Perform INSERT / UPDATE / DELETE / DDL.
  - Make gating decisions.
  - Build evidence packages.
  - Read from dashboard source tables (youtube_history, instagram_history, etc.).
"""
from __future__ import annotations

from typing import Optional

import psycopg
from psycopg_pool import AsyncConnectionPool

from app.core.config import get_settings
from app.core.exceptions import DataSourceConnectionError
from app.core.logging import get_logger

logger = get_logger(__name__)

# -- Platform configuration (AI tables only) --

_PLATFORM_CFG: dict[str, dict] = {
    "youtube": {
        "history_table": "youtube_history_ai",
        "content_table": "youtube_content_ai",
        "id_col": "video_id",
        "account_col": "channel_key",
        "primary_metric": "views",
    },
    "instagram": {
        "history_table": "instagram_history_ai",
        "content_table": "instagram_content_ai",
        "id_col": "media_id",
        "account_col": "account_key",
        "primary_metric": "reach",
    },
    "facebook": {
        "history_table": "facebook_history_ai",
        "content_table": "facebook_content_ai",
        "id_col": "post_id",
        "account_col": "account_key",
        "primary_metric": "reach",
    },
}

# -- Pool singleton --

_pool: Optional[AsyncConnectionPool] = None


async def open_pool() -> None:
    global _pool
    url = get_settings().DATABASE_URL
    if not url:
        logger.warning("DATABASE_URL is not configured — PostgreSQL data source disabled")
        return
    try:
        _pool = AsyncConnectionPool(url, min_size=1, max_size=10, open=False)
        await _pool.open()
        logger.info("PostgreSQL connection pool opened")
    except Exception as exc:
        _pool = None
        raise DataSourceConnectionError(f"Failed to open database pool: {exc}") from exc


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


def _get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise DataSourceConnectionError("Database pool is not initialised")
    return _pool


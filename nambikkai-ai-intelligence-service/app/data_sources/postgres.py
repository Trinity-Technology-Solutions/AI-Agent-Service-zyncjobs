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

from datetime import datetime
from typing import Optional

import psycopg
from psycopg_pool import AsyncConnectionPool

from app.core.config import get_settings
from app.core.exceptions import DataSourceConnectionError, DataSourceError
from app.core.logging import get_logger
from app.domain.models import HistorySnapshot, NormalizedContentRecord

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


# -- Internal query helpers --

async def _fetch_history(
    conn: psycopg.AsyncConnection,
    platform: str,
    content_id: str,
    hours: int,
) -> list[HistorySnapshot]:
    cfg = _PLATFORM_CFG[platform]
    table = cfg["history_table"]
    id_col = cfg["id_col"]
    metric_col = cfg["primary_metric"]

    sql = f"""
        SELECT collected_at, published_at, {metric_col}, likes, comments
        FROM {table}
        WHERE {id_col} = %s
          AND collected_at >= NOW() - INTERVAL '{hours} hours'
        ORDER BY collected_at ASC
    """  # table/column names are internal constants, not user input — safe
    async with conn.cursor() as cur:
        await cur.execute(sql, (content_id,))
        rows = await cur.fetchall()

    return [
        HistorySnapshot(
            collected_at=row[0],
            published_at=row[1],
            primary_metric_name=metric_col,
            primary_metric_value=row[2],
            likes=row[3],
            comments=row[4],
        )
        for row in rows
    ]


async def _fetch_content_youtube(
    conn: psycopg.AsyncConnection, video_id: str
) -> dict | None:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT channel_key, title, description, published_at,
                   category, language, creator_id, url
            FROM youtube_content_ai
            WHERE video_id = %s
            """,
            (video_id,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return {
        "account_key": row[0],
        "title": row[1],
        "description": row[2],
        "content_published_at": row[3],
        "category": row[4],
        "language": row[5],
        "creator_id": row[6],
        "url": row[7],
    }


async def _fetch_content_instagram(
    conn: psycopg.AsyncConnection, media_id: str
) -> dict | None:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT account_key, caption, published_at,
                   media_type, category, language, url
            FROM instagram_content_ai
            WHERE media_id = %s
            """,
            (media_id,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return {
        "account_key": row[0],
        "caption": row[1],
        "content_published_at": row[2],
        "media_type": row[3],
        "category": row[4],
        "language": row[5],
        "url": row[6],
    }


async def _fetch_content_facebook(
    conn: psycopg.AsyncConnection, post_id: str
) -> dict | None:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT account_key, caption, published_at,
                   post_type, category, language, url
            FROM facebook_content_ai
            WHERE post_id = %s
            """,
            (post_id,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return {
        "account_key": row[0],
        "caption": row[1],
        "content_published_at": row[2],
        "post_type": row[3],
        "category": row[4],
        "language": row[5],
        "url": row[6],
    }


_CONTENT_FETCHERS = {
    "youtube": _fetch_content_youtube,
    "instagram": _fetch_content_instagram,
    "facebook": _fetch_content_facebook,
}

# -- Public API --


async def fetch_normalized_record(
    platform: str,
    content_id: str,
    hours: int = 168,
) -> NormalizedContentRecord:
    """
    Fetch and join history + content for one content item from AI tables.

    Raises:
        ValueError: unknown platform.
        DataSourceConnectionError: pool not initialised or connection failed.
        DataSourceError: query failed.
    """
    if platform not in _PLATFORM_CFG:
        raise ValueError(f"Unknown platform: '{platform}'. Supported: {list(_PLATFORM_CFG)}")

    cfg = _PLATFORM_CFG[platform]
    pool = _get_pool()

    try:
        async with pool.connection() as conn:
            history = await _fetch_history(conn, platform, content_id, hours)
            content = await _CONTENT_FETCHERS[platform](conn, content_id)
    except DataSourceConnectionError:
        raise
    except Exception as exc:
        raise DataSourceError(f"Query failed for {platform}/{content_id}: {exc}") from exc

    content_extra: dict = {}
    if content is not None:
        content_extra = dict(content)
        account_key = content_extra.pop("account_key", None) or ""
    else:
        account_key = ""
        if history:
            account_key = await _fetch_account_key_from_history(platform, content_id)

    return NormalizedContentRecord(
        platform=platform,
        content_id=content_id,
        account_key=account_key,
        primary_metric_name=cfg["primary_metric"],
        history=history,
        **content_extra,
    )


async def _fetch_account_key_from_history(platform: str, content_id: str) -> str:
    cfg = _PLATFORM_CFG[platform]
    table = cfg["history_table"]
    id_col = cfg["id_col"]
    account_col = cfg["account_col"]
    pool = _get_pool()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"SELECT {account_col} FROM {table} WHERE {id_col} = %s LIMIT 1",
                    (content_id,),
                )
                row = await cur.fetchone()
        return row[0] if row else ""
    except Exception:
        return ""


async def fetch_all_content_ids(platform: str) -> list[str]:
    """
    Return all distinct content IDs that have history rows in the AI tables.
    """
    if platform not in _PLATFORM_CFG:
        raise ValueError(f"Unknown platform: '{platform}'")

    cfg = _PLATFORM_CFG[platform]
    table = cfg["history_table"]
    id_col = cfg["id_col"]
    pool = _get_pool()

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"SELECT DISTINCT {id_col} FROM {table}")
                rows = await cur.fetchall()
        return [row[0] for row in rows]
    except Exception as exc:
        raise DataSourceError(f"Failed to list content IDs for {platform}: {exc}") from exc

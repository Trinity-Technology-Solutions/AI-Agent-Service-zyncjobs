"""
Dashboard historical-data API client.

Responsibilities:
  - Authenticate with x-api-key header (never log the key).
  - Paginate through all pages until exhausted.
  - Validate response shape before returning data.
  - Normalise external API rows into internal HistoryRow / ContentRow types.
  - Raise clear exceptions on HTTP failures or API-level errors.

This module must NOT:
  - Write to the database.
  - Make gating decisions.
  - Call an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import DataSourceError
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_PLATFORMS = ("youtube", "instagram", "facebook")


# ---------------------------------------------------------------------------
# Internal normalised types
# ---------------------------------------------------------------------------

@dataclass
class HistoryRow:
    platform: str
    content_id: str          # video_id / media_id / post_id
    account_key: str         # channel_key / account_key
    collected_at: datetime
    published_at: datetime
    primary_metric: int      # views (YouTube) or reach (Instagram/Facebook)
    likes: int
    comments: int


@dataclass
class ContentRow:
    platform: str
    content_id: str
    account_key: str
    title: Optional[str] = None        # YouTube
    caption: Optional[str] = None      # Instagram / Facebook
    description: Optional[str] = None  # YouTube
    published_at: Optional[datetime] = None
    category: Optional[str] = None
    language: Optional[str] = None
    creator_id: Optional[str] = None   # YouTube
    media_type: Optional[str] = None   # Instagram
    post_type: Optional[str] = None    # Facebook
    url: Optional[str] = None
    resolution: Optional[str] = None
    transcript: Optional[str] = None
    transcript_status: Optional[str] = None


@dataclass
class IngestionPage:
    history: list[HistoryRow] = field(default_factory=list)
    content: list[ContentRow] = field(default_factory=list)
    offset: int = 0
    limit: int = 0


# ---------------------------------------------------------------------------
# Platform field mapping
# ---------------------------------------------------------------------------

_ID_FIELD = {
    "youtube": "video_id",
    "instagram": "media_id",
    "facebook": "post_id",
}

_ACCOUNT_FIELD = {
    "youtube": "channel_key",
    "instagram": "account_key",
    "facebook": "account_key",
}

_PRIMARY_METRIC_FIELD = {
    "youtube": "views",
    "instagram": "reach",
    "facebook": "reach",
}


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _normalise_history_row(platform: str, raw: dict) -> Optional[HistoryRow]:
    id_field = _ID_FIELD[platform]
    account_field = _ACCOUNT_FIELD[platform]
    metric_field = _PRIMARY_METRIC_FIELD[platform]

    content_id = raw.get(id_field)
    account_key = raw.get(account_field)
    collected_at = _parse_dt(raw.get("collected_at"))
    published_at = _parse_dt(raw.get("published_at"))

    if not content_id or not collected_at or not published_at:
        return None

    return HistoryRow(
        platform=platform,
        content_id=str(content_id),
        account_key=str(account_key or ""),
        collected_at=collected_at,
        published_at=published_at,
        primary_metric=int(raw.get(metric_field) or 0),
        likes=int(raw.get("likes") or 0),
        comments=int(raw.get("comments") or 0),
    )


def _normalise_content_row(platform: str, raw: dict) -> Optional[ContentRow]:
    id_field = _ID_FIELD[platform]
    account_field = _ACCOUNT_FIELD[platform]

    content_id = raw.get(id_field)
    if not content_id:
        return None

    return ContentRow(
        platform=platform,
        content_id=str(content_id),
        account_key=str(raw.get(account_field) or ""),
        title=raw.get("title"),
        caption=raw.get("caption"),
        description=raw.get("description"),
        published_at=_parse_dt(raw.get("published_at")),
        category=raw.get("category"),
        language=raw.get("language"),
        creator_id=raw.get("creator_id"),
        media_type=raw.get("media_type"),
        post_type=raw.get("post_type"),
        url=raw.get("url"),
        resolution=raw.get("resolution"),
        transcript=raw.get("transcript"),
        transcript_status=raw.get("transcript_status"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_historical_page(
    start_date: datetime,
    end_date: datetime,
    platforms: tuple[str, ...] = SUPPORTED_PLATFORMS,
    offset: int = 0,
) -> IngestionPage:
    """
    Fetch one page of historical data from the dashboard API.

    Raises:
        DataSourceError: on HTTP failure, API-level error, or malformed response.
    """
    settings = get_settings()

    if not settings.NAMBIKKAI_API_URL:
        raise DataSourceError("NAMBIKKAI_API_URL is not configured")
    if not settings.AI_AGENT_API_KEY:
        raise DataSourceError("AI_AGENT_API_KEY is not configured")

    params = {
        "platforms": ",".join(platforms),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "limit": settings.NAMBIKKAI_API_PAGE_SIZE,
        "offset": offset,
    }

    timeout = httpx.Timeout(
        connect=10.0,
        read=float(settings.NAMBIKKAI_API_TIMEOUT_SECONDS),
        write=10.0,
        pool=10.0,
    )

    logger.info(
        "Fetching historical data page",
        extra={"offset": offset, "limit": settings.NAMBIKKAI_API_PAGE_SIZE, "platforms": platforms},
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                settings.NAMBIKKAI_API_URL,
                headers={"x-api-key": settings.AI_AGENT_API_KEY},
                params=params,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise DataSourceError(
            f"Dashboard API HTTP {exc.response.status_code} at offset={offset}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise DataSourceError(
            f"Dashboard API timed out at offset={offset}"
        ) from exc
    except httpx.RequestError as exc:
        raise DataSourceError(
            f"Dashboard API request error at offset={offset}: {type(exc).__name__}"
        ) from exc

    try:
        body = response.json()
    except Exception as exc:
        raise DataSourceError(f"Dashboard API returned non-JSON response at offset={offset}") from exc

    if not body.get("ok"):
        raise DataSourceError(f"Dashboard API returned ok=false at offset={offset}: {body.get('error')}")

    raw_data = body.get("data")
    if not isinstance(raw_data, dict):
        raise DataSourceError(f"Dashboard API response missing 'data' dict at offset={offset}")

    pagination = body.get("pagination", {})
    actual_limit = int(pagination.get("limit", settings.NAMBIKKAI_API_PAGE_SIZE))

    page = IngestionPage(offset=offset, limit=actual_limit)

    for platform in platforms:
        platform_data = raw_data.get(platform)
        if not isinstance(platform_data, dict):
            continue

        for raw_row in platform_data.get("history") or []:
            row = _normalise_history_row(platform, raw_row)
            if row is not None:
                page.history.append(row)

        for raw_row in platform_data.get("content") or []:
            row = _normalise_content_row(platform, raw_row)
            if row is not None:
                page.content.append(row)

    logger.info(
        "Historical data page received",
        extra={
            "offset": offset,
            "history_rows": len(page.history),
            "content_rows": len(page.content),
            "actual_limit": actual_limit,
        },
    )

    return page


async def fetch_all_historical(
    start_date: datetime,
    end_date: datetime,
    platforms: tuple[str, ...] = SUPPORTED_PLATFORMS,
) -> tuple[list[HistoryRow], list[ContentRow]]:
    """
    Paginate through all pages and return the complete history + content sets.

    Pagination stops when a page returns fewer history rows than the API's
    actual limit (not the requested page size — the API may cap it lower).
    """
    all_history: list[HistoryRow] = []
    all_content: list[ContentRow] = []
    offset = 0
    page_num = 0

    while True:
        page_num += 1
        page = await fetch_historical_page(
            start_date=start_date,
            end_date=end_date,
            platforms=platforms,
            offset=offset,
        )

        all_history.extend(page.history)
        all_content.extend(page.content)

        logger.info(
            "Pagination progress",
            extra={
                "page": page_num,
                "offset": offset,
                "page_history": len(page.history),
                "total_history_so_far": len(all_history),
            },
        )

        # Stop when the page returned fewer rows than the API's actual limit.
        # This correctly handles the case where the API caps page size lower
        # than what was requested (e.g. requested 500, API returns limit=50).
        if len(page.history) < page.limit:
            break

        offset += page.limit

    return all_history, all_content

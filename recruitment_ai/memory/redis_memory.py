"""Redis-backed memory storage — conversation history + user data.
Reuses the cache_service Redis client to avoid duplicate connections.
Gracefully degrades: all operations are no-ops if Redis is unavailable.
"""
import json
import logging
from typing import Optional
from recruitment_ai.services.cache_service import cache_service

logger = logging.getLogger(__name__)

MEMORY_TTL = 60 * 60 * 2       # 2 hours
MEMORY_WINDOW = 10             # max conversation turns


def _conversation_key(session_id: str) -> str:
    return f"zyncjobs:memory:conv:{session_id}"


def _user_key(user_id: str) -> str:
    return f"zyncjobs:memory:user:{user_id}"


class RedisMemory:
    """Low-level Redis operations for memory. All methods are safe to call when Redis is down."""

    @property
    def _ready(self) -> bool:
        return cache_service._available and cache_service._client is not None

    # ── Conversation ───────────────────────────────────────────────────────

    async def get_conversation(self, session_id: str) -> list[dict]:
        """Return full conversation history for a session. Returns [] on miss or error."""
        if not self._ready:
            return []
        try:
            key = _conversation_key(session_id)
            raw = await cache_service._client.get(key)
            return json.loads(raw) if raw else []
        except Exception as e:
            logger.warning("Redis get_conversation error: %s", e)
            return []

    async def set_conversation(self, session_id: str, turns: list[dict]) -> None:
        """Store conversation history, trimmed to MEMORY_WINDOW * 2 messages."""
        if not self._ready:
            return
        try:
            key = _conversation_key(session_id)
            trimmed = turns[-(MEMORY_WINDOW * 2):]
            await cache_service._client.setex(key, MEMORY_TTL, json.dumps(trimmed))
        except Exception as e:
            logger.warning("Redis set_conversation error: %s", e)

    async def append_conversation(self, session_id: str, new_turns: list[dict]) -> None:
        """Append turns to existing conversation history, then trim and persist."""
        if not self._ready:
            return
        try:
            existing = await self.get_conversation(session_id)
            existing.extend(new_turns)
            await self.set_conversation(session_id, existing)
        except Exception as e:
            logger.warning("Redis append_conversation error: %s", e)

    # ── User data ──────────────────────────────────────────────────────────

    async def get_user_data(self, user_id: str) -> dict:
        """Return user preferences / profile data. Returns {} on miss or error."""
        if not self._ready:
            return {}
        try:
            key = _user_key(user_id)
            raw = await cache_service._client.get(key)
            return json.loads(raw) if raw else {}
        except Exception as e:
            logger.warning("Redis get_user_data error: %s", e)
            return {}

    async def set_user_data(self, user_id: str, data: dict) -> None:
        """Store user data with standard TTL."""
        if not self._ready:
            return
        try:
            key = _user_key(user_id)
            await cache_service._client.setex(key, MEMORY_TTL, json.dumps(data))
        except Exception as e:
            logger.warning("Redis set_user_data error: %s", e)


redis_memory = RedisMemory()

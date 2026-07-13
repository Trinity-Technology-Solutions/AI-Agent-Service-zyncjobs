"""Redis cache service — caches LLM results by (intent + query) hash.
Gracefully degrades: if Redis is unavailable, all operations are no-ops.
Architecture doc: Context Retrieval (Qdrant + PostgreSQL + Redis).
"""
import hashlib
import json
import logging
from typing import Optional
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)

# TTL per intent — assessment/roadmap results cached longer, chat shorter
INTENT_TTL: dict[str, int] = {
    "SKILL_ASSESSMENT": 3600,   # 1 hour — questions don't change often
    "CAREER_ADVICE": 1800,      # 30 min
    "JD_GENERATOR": 1800,       # 30 min
    "JOB_PARSER": 900,          # 15 min
    "ATS_SCORE": 900,
    "JOB_MATCH": 600,
    "RESUME_PARSER": 600,
    "INTERVIEW_PREP": 1800,
    "RESUME_BUILDER": 600,
    "RECRUITER": 300,
    "CHAT": 120,                # 2 min — conversational, keep fresh
}
DEFAULT_TTL = int(settings.__dict__.get("CONVERSATION_TTL", 3600))


class CacheService:
    """Redis-backed result cache with silent fallback."""

    def __init__(self):
        self._client = None
        self._available = False

    async def connect(self) -> None:
        """Try to connect to Redis. Failure is non-fatal."""
        redis_url = getattr(settings, "REDIS_URL", "") or ""
        if not redis_url:
            logger.info("Redis URL not configured — cache disabled")
            return
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2)
            await self._client.ping()
            self._available = True
            logger.info("Redis cache connected: %s", redis_url)
        except Exception as e:
            self._available = False
            self._client = None
            logger.warning("Redis unavailable — running without cache: %s", e)

    def _make_key(self, intent: str, query: str) -> str:
        raw = f"{intent}:{query.strip().lower()}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"zyncjobs:ai:{intent.lower()}:{digest}"

    async def get(self, intent: str, query: str) -> Optional[dict]:
        """Return cached result or None."""
        if not self._available or not self._client:
            return None
        try:
            key = self._make_key(intent, query)
            value = await self._client.get(key)
            if value:
                logger.debug("Cache HIT: %s", key)
                return json.loads(value)
        except Exception as e:
            logger.warning("Cache get error: %s", e)
        return None

    async def set(self, intent: str, query: str, result: dict) -> None:
        """Store result in cache with intent-specific TTL."""
        if not self._available or not self._client:
            return
        try:
            key = self._make_key(intent, query)
            ttl = INTENT_TTL.get(intent, DEFAULT_TTL)
            await self._client.setex(key, ttl, json.dumps(result))
            logger.debug("Cache SET: %s (ttl=%ds)", key, ttl)
        except Exception as e:
            logger.warning("Cache set error: %s", e)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
            self._available = False


cache_service = CacheService()

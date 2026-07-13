"""Session memory service — persists conversation history per session_id.
Architecture doc: Load Memory → (brains use history) → Store Memory.
Storage: Redis (fast, TTL-based) + SQLite/PostgreSQL (persistent).
Gracefully degrades if DB or Redis is unavailable.
"""
import json
import logging
from typing import Optional
from recruitment_ai.services.cache_service import cache_service

logger = logging.getLogger(__name__)

MEMORY_WINDOW = 10        # max turns to load into context
MEMORY_TTL = 60 * 60 * 2  # 2 hours in Redis
CHAT_INTENTS = {"CHAT", "CAREER_ADVICE", "RECRUITER"}  # only these benefit from history


def _redis_key(session_id: str) -> str:
    return f"zyncjobs:memory:{session_id}"


class MemoryService:
    """Loads and stores conversation turns per session."""

    async def load(self, session_id: Optional[str]) -> list[dict]:
        """Return last MEMORY_WINDOW turns for this session. Returns [] on any failure."""
        if not session_id:
            return []

        # 1. Try Redis first (fast path)
        history = await self._load_from_redis(session_id)
        if history:
            return history

        # 2. Fall back to DB
        return await self._load_from_db(session_id)

    async def store(self, session_id: Optional[str], user_id: Optional[str], query: str, result: dict, intent: str) -> None:
        """Persist the current turn (user query + assistant reply)."""
        if not session_id or not query:
            return

        reply = (
            result.get("reply") or
            result.get("advice") or
            result.get("search_strategy") or
            json.dumps(result)[:500]
        )

        turn = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": reply, "intent": intent},
        ]

        await self._store_to_redis(session_id, turn)
        await self._store_to_db(session_id, user_id, turn, intent)

    # ── Redis ──────────────────────────────────────────────────────────────

    async def _load_from_redis(self, session_id: str) -> list[dict]:
        if not cache_service._available or not cache_service._client:
            return []
        try:
            key = _redis_key(session_id)
            raw = await cache_service._client.get(key)
            if raw:
                return json.loads(raw)[-MEMORY_WINDOW * 2:]  # each turn = 2 messages
        except Exception as e:
            logger.warning("Memory Redis load error: %s", e)
        return []

    async def _store_to_redis(self, session_id: str, new_turns: list[dict]) -> None:
        if not cache_service._available or not cache_service._client:
            return
        try:
            key = _redis_key(session_id)
            raw = await cache_service._client.get(key)
            history: list = json.loads(raw) if raw else []
            history.extend(new_turns)
            # Keep only last MEMORY_WINDOW * 2 messages
            history = history[-(MEMORY_WINDOW * 2):]
            await cache_service._client.setex(key, MEMORY_TTL, json.dumps(history))
        except Exception as e:
            logger.warning("Memory Redis store error: %s", e)

    # ── Database ───────────────────────────────────────────────────────────

    async def _load_from_db(self, session_id: str) -> list[dict]:
        try:
            from recruitment_ai.database.connection import get_db
            from recruitment_ai.database.models import Conversation
            from sqlalchemy import select, desc

            async for db in get_db():
                stmt = (
                    select(Conversation)
                    .where(Conversation.session_id == session_id)
                    .order_by(desc(Conversation.created_at))
                    .limit(MEMORY_WINDOW * 2)
                )
                rows = (await db.execute(stmt)).scalars().all()
                rows = list(reversed(rows))
                return [{"role": r.role, "content": r.content} for r in rows]
        except Exception as e:
            logger.warning("Memory DB load error: %s", e)
        return []

    async def _store_to_db(self, session_id: str, user_id: Optional[str], turns: list[dict], intent: str) -> None:
        try:
            from recruitment_ai.database.connection import get_db
            from recruitment_ai.database.models import Conversation

            async for db in get_db():
                for turn in turns:
                    db.add(Conversation(
                        session_id=session_id,
                        role=turn["role"],
                        content=turn["content"],
                        metadata_json={"intent": intent},
                    ))
        except Exception as e:
            logger.warning("Memory DB store error: %s", e)


memory_service = MemoryService()

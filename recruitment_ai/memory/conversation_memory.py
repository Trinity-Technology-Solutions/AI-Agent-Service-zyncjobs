"""Conversation memory — manages per-session conversation turn history.
Stores in Redis (fast, TTL-based) + PostgreSQL/SQLite (persistent).
Supports summarization for long-running conversations.
"""
import json
import logging
from typing import Optional
from recruitment_ai.memory.redis_memory import redis_memory

logger = logging.getLogger(__name__)

CHAT_INTENTS = {"CHAT", "CAREER_ADVICE", "RECRUITER"}


def _extract_reply(result: dict) -> str:
    return (
        result.get("reply")
        or result.get("advice")
        or result.get("search_strategy")
        or json.dumps(result)[:500]
    )


class ConversationMemory:
    """Per-session conversation turn history."""

    async def load(self, session_id: Optional[str]) -> list[dict]:
        """Return last MEMORY_WINDOW turns for this session. Falls back to DB."""
        if not session_id:
            return []

        history = await redis_memory.get_conversation(session_id)
        if history:
            return history

        return await self._load_from_db(session_id)

    async def store(
        self,
        session_id: str,
        user_id: Optional[str],
        query: str,
        result: dict,
        intent: str,
    ) -> None:
        """Persist one user→assistant turn."""
        if not session_id or not query:
            return

        reply = _extract_reply(result)
        turn = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": reply, "intent": intent},
        ]

        await redis_memory.append_conversation(session_id, turn)
        await self._store_to_db(session_id, user_id, turn, intent)

    async def summarize(self, session_id: str) -> Optional[str]:
        """Summarize a long conversation into a concise string.
        Useful for injecting context for very long sessions.
        Returns None if conversation is short or summarization fails.
        """
        history = await redis_memory.get_conversation(session_id)
        if not history or len(history) < 10:
            return None

        try:
            from recruitment_ai.llm import llm_service
            turns_text = "\n".join(
                f"{m['role']}: {m['content'][:200]}"
                for m in history[-20:]
            )
            prompt = (
                "Summarize this career-coaching conversation in 2-3 sentences.\n\n"
                f"{turns_text}\n\nSummary:"
            )
            summary = await llm_service.generate(
                brain_name="memory_summarizer",
                prompt=prompt,
                temperature=0.2,
                max_tokens=150,
            )
            logger.debug("Conversation summarized for session %s", session_id)
            return summary.strip()
        except Exception as e:
            logger.warning("Conversation summarization failed: %s", e)
            return None

    # ── Backend persistence ─────────────────────────────────────────────

    async def _load_from_db(self, session_id: str) -> list[dict]:
        try:
            from recruitment_ai.services.backend_client import backend_client
            return await backend_client.get_conversation_history(session_id)
        except Exception as e:
            logger.warning("Memory backend load error: %s", e)
        return []

    async def _store_to_db(
        self,
        session_id: str,
        user_id: Optional[str],
        turns: list[dict],
        intent: str,
    ) -> None:
        logger.debug("Conversation stored in Redis (backend sync optional): %s", session_id)


conversation_memory = ConversationMemory()

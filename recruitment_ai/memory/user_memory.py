"""User memory — preferences, profile data, and cross-session history.
Enables personalized responses based on user's stored data.
"""
import logging
from typing import Optional
from recruitment_ai.memory.redis_memory import redis_memory

logger = logging.getLogger(__name__)


class UserMemory:
    """Per-user memory aggregated across sessions."""

    async def get_preferences(self, user_id: Optional[str]) -> dict:
        """Return stored preferences for a user. Returns {} on miss."""
        if not user_id:
            return {}
        return await redis_memory.get_user_data(user_id)

    async def set_preference(self, user_id: str, key: str, value) -> None:
        """Store a single preference key-value pair."""
        if not user_id:
            return
        data = await redis_memory.get_user_data(user_id)
        data[key] = value
        await redis_memory.set_user_data(user_id, data)

    async def update_preferences(self, user_id: str, updates: dict) -> None:
        """Merge updates into existing preferences."""
        if not user_id:
            return
        data = await redis_memory.get_user_data(user_id)
        data.update(updates)
        await redis_memory.set_user_data(user_id, data)

    async def get_cross_session_history(self, user_id: str, limit: int = 5) -> list[dict]:
        """Load recent conversations across all of a user's sessions from backend."""
        if not user_id:
            return []
        try:
            from recruitment_ai.services.backend_client import backend_client
            return await backend_client.get_conversation_history(user_id)
        except Exception as e:
            logger.warning("User cross-session history error: %s", e)
        return []


user_memory = UserMemory()

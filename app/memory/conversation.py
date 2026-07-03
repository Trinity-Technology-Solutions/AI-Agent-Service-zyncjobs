import json
from datetime import datetime, timezone
from app.config.settings import settings
from .backend import memory_backend


def _key(user_id: str) -> str:
    return f"conv:{user_id}"


class ConversationMemory:
    def add(self, user_id: str, entry: dict):
        raw = memory_backend.get(_key(user_id))
        history: list[dict] = json.loads(raw) if raw else []
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        history.append(entry)
        window = settings.AGENT_MEMORY_WINDOW * 2
        if len(history) > window:
            history = history[-window:]
        memory_backend.set(_key(user_id), json.dumps(history), ttl=settings.CONVERSATION_TTL)

    def get_history(self, user_id: str, limit: int = 0) -> list[dict]:
        raw = memory_backend.get(_key(user_id))
        history: list[dict] = json.loads(raw) if raw else []
        effective_limit = limit or settings.AGENT_MEMORY_WINDOW * 2
        return history[-effective_limit:]

    def clear(self, user_id: str):
        memory_backend.delete(_key(user_id))

"""Persistent memory backend: Redis if configured, else in-memory fallback."""
import json
import time
from typing import Optional
from app.config.settings import settings


class MemoryBackend:
    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, ttl: int = 0): ...
    def delete(self, key: str): ...
    def exists(self, key: str) -> bool: ...
    def close(self): ...


class InMemoryBackend(MemoryBackend):
    def __init__(self):
        self._data: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> Optional[str]:
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, val = entry
        if ts and time.time() > ts:
            del self._data[key]
            return None
        return val

    def set(self, key: str, value: str, ttl: int = 0):
        expires = time.time() + ttl if ttl > 0 else 0
        self._data[key] = (expires, value)

    def delete(self, key: str):
        self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def close(self):
        self._data.clear()


class RedisBackend(MemoryBackend):
    def __init__(self):
        import redis as _redis
        self._client = _redis.Redis.from_url(settings.REDIS_URL)

    def get(self, key: str) -> Optional[str]:
        val = self._client.get(key)
        return val.decode() if val else None

    def set(self, key: str, value: str, ttl: int = 0):
        if ttl > 0:
            self._client.setex(key, ttl, value)
        else:
            self._client.set(key, value)

    def delete(self, key: str):
        self._client.delete(key)

    def exists(self, key: str) -> bool:
        return bool(self._client.exists(key))

    def close(self):
        self._client.close()


def create_backend() -> MemoryBackend:
    if settings.REDIS_URL:
        try:
            return RedisBackend()
        except Exception:
            pass
    return InMemoryBackend()


memory_backend: MemoryBackend = create_backend()
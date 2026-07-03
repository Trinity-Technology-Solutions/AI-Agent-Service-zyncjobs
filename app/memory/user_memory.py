import json
from datetime import datetime, timezone
from .backend import memory_backend


def _key(user_id: str) -> str:
    return f"user:{user_id}"


def _subkey(user_id: str, field: str) -> str:
    return f"user:{user_id}:{field}"


class UserMemory:
    def save_profile(self, user_id: str, data: dict):
        raw = memory_backend.get(_subkey(user_id, "profile"))
        profile: dict = json.loads(raw) if raw else {}
        profile.update(data)
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        memory_backend.set(_subkey(user_id, "profile"), json.dumps(profile))

    def get_profile(self, user_id: str) -> dict:
        raw = memory_backend.get(_subkey(user_id, "profile"))
        return json.loads(raw) if raw else {}

    def save_resume(self, user_id: str, resume_data: dict):
        resume_data["saved_at"] = datetime.now(timezone.utc).isoformat()
        memory_backend.set(_subkey(user_id, "resume"), json.dumps(resume_data))

    def get_resume(self, user_id: str) -> dict:
        raw = memory_backend.get(_subkey(user_id, "resume"))
        return json.loads(raw) if raw else {}

    def save_skills(self, user_id: str, skills: list[str]):
        raw = memory_backend.get(_subkey(user_id, "skills"))
        existing: list[str] = json.loads(raw) if raw else []
        merged = list(dict.fromkeys(existing + skills))
        memory_backend.set(_subkey(user_id, "skills"), json.dumps(merged))

    def get_skills(self, user_id: str) -> list[str]:
        raw = memory_backend.get(_subkey(user_id, "skills"))
        return json.loads(raw) if raw else []

    def save_goal(self, user_id: str, goal: str):
        data = {"goal": goal, "set_at": datetime.now(timezone.utc).isoformat()}
        memory_backend.set(_subkey(user_id, "goal"), json.dumps(data))

    def get_goal(self, user_id: str) -> str:
        raw = memory_backend.get(_subkey(user_id, "goal"))
        return json.loads(raw).get("goal", "") if raw else ""

    def remember(self, user_id: str, key: str, value):
        memory_backend.set(_subkey(user_id, key), json.dumps(value))

    def recall(self, user_id: str, key: str):
        raw = memory_backend.get(_subkey(user_id, key))
        return json.loads(raw) if raw else None

    def clear(self, user_id: str):
        memory_backend.delete(_key(user_id))
        for field in ("profile", "resume", "skills", "goal"):
            memory_backend.delete(_subkey(user_id, field))
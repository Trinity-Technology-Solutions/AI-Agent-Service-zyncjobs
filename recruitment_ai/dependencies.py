"""FastAPI dependency injection hub — matches roadmap structure."""
from recruitment_ai.auth.jwt_handler import get_current_user, create_access_token
from recruitment_ai.config.settings import settings, get_settings

__all__ = ["get_current_user", "create_access_token", "settings", "get_settings"]

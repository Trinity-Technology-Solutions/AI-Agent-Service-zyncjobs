"""Per-brain model configuration — matches roadmap spec."""
from recruitment_ai.config.settings import settings


def get_model_for_brain(brain_name: str) -> str:
    return settings.OLLAMA_MODELS.get(brain_name.lower(), "qwen2.5:3b")


def get_all_models() -> dict:
    return dict(settings.OLLAMA_MODELS)

from app.core.config import get_settings
from app.providers.base import LLMProvider


def get_provider() -> LLMProvider:
    provider_name = get_settings().AI_PROVIDER.lower()
    if provider_name == "lmstudio":
        from app.providers.lmstudio import LMStudioProvider
        return LMStudioProvider()
    if provider_name == "bedrock":
        from app.providers.bedrock import BedrockProvider
        return BedrockProvider()
    raise ValueError(f"Unknown AI_PROVIDER: '{provider_name}'. Supported: lmstudio, bedrock")

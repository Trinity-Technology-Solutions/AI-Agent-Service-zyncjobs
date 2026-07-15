"""LLM factory — returns the correct provider based on LLM_PROVIDER env var.

Development  → LLM_PROVIDER=ollama  → OllamaProvider (local Ollama)
QA           → LLM_PROVIDER=bedrock → BedrockProvider (Claude Haiku on Bedrock)
Production   → LLM_PROVIDER=bedrock → BedrockProvider (Claude Sonnet on Bedrock)
"""
from functools import lru_cache
from recruitment_ai.services.llm.base import BaseLLMProvider
from recruitment_ai.config.settings import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseLLMProvider:
    if settings.LLM_PROVIDER.lower() == "bedrock":
        from recruitment_ai.services.llm.bedrock_provider import BedrockProvider
        return BedrockProvider()
    from recruitment_ai.services.llm.ollama_provider import OllamaProvider
    return OllamaProvider()

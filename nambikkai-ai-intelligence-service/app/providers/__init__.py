"""
LLM provider factory with optional fallback support.

Provider selection is driven entirely by configuration (AI_PROVIDER env var).
No production domain names, URLs, or credentials are hardcoded here.

Fallback behaviour:
  If LLM_FALLBACK_PROVIDER is set in config to a non-empty value that differs
  from AI_PROVIDER, get_provider_with_fallback() will attempt the primary
  provider first and silently switch to the fallback on ProviderUnavailableError.

  The fallback is NOT used for ProviderError (bad model output, validation
  failures, etc.) — those are the model's fault, not the provider's availability.

  Fallback is disabled by default (LLM_FALLBACK_PROVIDER is blank).
  Never enable fallback from production Bedrock → local LM Studio because
  local LM Studio is not reachable from a production EC2 instance.

Provider metadata always records:
  - which provider was selected
  - which provider actually served the response
  - whether fallback occurred
  - why the primary failed (if fallback occurred)
"""
from __future__ import annotations

import logging
from typing import Optional

from app.core.config import get_settings
from app.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def _make_provider(name: str) -> LLMProvider:
    """Instantiate a provider by name. Raises ValueError for unknown names."""
    n = name.lower().strip()
    if n == "lmstudio":
        from app.providers.lmstudio import LMStudioProvider
        return LMStudioProvider()
    if n == "bedrock":
        from app.providers.bedrock import BedrockProvider
        return BedrockProvider()
    raise ValueError(f"Unknown AI provider: '{name}'. Supported: lmstudio, bedrock")


def get_provider() -> LLMProvider:
    """Return the configured primary LLM provider."""
    return _make_provider(get_settings().AI_PROVIDER)


def get_provider_with_fallback() -> tuple[LLMProvider, Optional[LLMProvider]]:
    """
    Return (primary_provider, fallback_provider_or_None).

    The fallback is only created when LLM_FALLBACK_PROVIDER is configured and
    differs from AI_PROVIDER.  The caller decides when to use the fallback.
    """
    settings = get_settings()
    primary = _make_provider(settings.AI_PROVIDER)

    fallback_name = getattr(settings, "LLM_FALLBACK_PROVIDER", "").strip()
    if not fallback_name or fallback_name.lower() == settings.AI_PROVIDER.lower():
        return primary, None

    try:
        fallback = _make_provider(fallback_name)
        logger.info(
            "[Providers] Primary=%s Fallback=%s",
            settings.AI_PROVIDER,
            fallback_name,
        )
        return primary, fallback
    except ValueError as exc:
        logger.warning("[Providers] Fallback provider config invalid: %s", exc)
        return primary, None

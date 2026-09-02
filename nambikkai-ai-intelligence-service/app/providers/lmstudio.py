import json
import time
import uuid
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import ProviderError, ProviderUnavailableError
from app.core.logging import get_logger
from app.domain.models import EditorialAnalysis, EvidencePackage
from app.providers.base import LLMProvider

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are an editorial analytics assistant. "
    "Return ONLY a valid JSON object — no markdown, no code fences, no explanation, no trailing text. "
    "The JSON object must contain exactly these keys: "
    "content_intent (string), "
    "observed_signals (array of strings, min 1), "
    "possible_contributing_factors (array of strings, min 1), "
    "writer_recommendations (array of strings, min 1), "
    "keyword_suggestions (array of strings), "
    "title_suggestions (array of strings), "
    "description_suggestions (array of strings), "
    "hashtag_suggestions (array of strings), "
    "cross_platform_ideas (array of strings), "
    "confidence (float between 0.0 and 1.0), "
    "limitations (array of strings). "
    "Use 'possible contributing factor' or 'observed signal' language — never assert causation. "
    "Do not invent metrics. "
    "All string values must use valid JSON escaping. "
    "No trailing commas. Use null for missing values, not Python None. "
    "Output the JSON object and nothing else."
)


def _build_user_prompt(evidence: EvidencePackage) -> str:
    lines = [
        f"Content: {evidence.content_metadata.title}",
        f"Platform: {evidence.content_metadata.platform or 'unknown'}",
        f"Gate: {evidence.gate_result.classification.value}",
        f"Velocity ratio: {evidence.gate_result.velocity_ratio}",
        f"Like acceleration: {evidence.gate_result.like_acceleration}",
        f"Reason: {evidence.gate_result.reason}",
        f"Baseline available: {evidence.data_quality.baseline_available}",
    ]
    if evidence.transcript_excerpt:
        lines.append(f"Excerpt: {evidence.transcript_excerpt}")
    if evidence.regional_signals:
        lines.append(f"Regional: {evidence.regional_signals}")
    lines.append("Return the JSON object now.")
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s[: s.rfind("```")]
    return s.strip()


class LMStudioProvider(LLMProvider):

    def __init__(self) -> None:
        self._settings = get_settings()

    async def generate_structured_analysis(self, evidence: EvidencePackage) -> EditorialAnalysis:
        settings = self._settings
        request_id = uuid.uuid4().hex[:12]
        user_prompt = _build_user_prompt(evidence)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": settings.LMSTUDIO_MODEL,
            "messages": messages,
            "temperature": settings.LMSTUDIO_TEMPERATURE,
            "max_tokens": settings.LMSTUDIO_MAX_TOKENS,
        }
        endpoint = f"{settings.LMSTUDIO_BASE_URL}/chat/completions"
        input_chars = len(_SYSTEM_PROMPT) + len(user_prompt)

        logger.info(
            "LMStudio request starting",
            extra={
                "request_id": request_id,
                "provider": "lmstudio",
                "model": settings.LMSTUDIO_MODEL,
                "endpoint": endpoint,
                "configured_timeout_s": settings.LMSTUDIO_TIMEOUT_SECONDS,
                "max_tokens": settings.LMSTUDIO_MAX_TOKENS,
                "temperature": settings.LMSTUDIO_TEMPERATURE,
                "num_messages": len(messages),
                "input_chars": input_chars,
            },
        )

        timeout = httpx.Timeout(
            connect=10.0,
            read=float(settings.LMSTUDIO_TIMEOUT_SECONDS),
            write=10.0,
            pool=10.0,
        )

        t_start = time.monotonic()
        response = None
        try:
            # IMPORTANT: raise_for_status() must be called INSIDE the async with block.
            # Calling it after the block exits closes the connection before the response
            # body can be read, causing Channel Error on non-2xx responses.
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Cannot connect to LM Studio at {settings.LMSTUDIO_BASE_URL}"
            ) from exc
        except httpx.TimeoutException as exc:
            duration_ms = int((time.monotonic() - t_start) * 1000)
            logger.error(
                "LMStudio request timed out",
                extra={"request_id": request_id, "duration_ms": duration_ms},
            )
            raise ProviderUnavailableError(
                f"LM Studio timed out after {duration_ms}ms "
                f"(configured read timeout: {settings.LMSTUDIO_TIMEOUT_SECONDS}s)"
            ) from exc
        except httpx.HTTPStatusError as exc:
            duration_ms = int((time.monotonic() - t_start) * 1000)
            # Safely capture the response body for diagnostics — it is still readable
            # here because raise_for_status() is now called inside the async with block.
            try:
                error_body = exc.response.text[:500]
            except Exception:
                error_body = "<unreadable>"
            logger.error(
                "LMStudio HTTP error",
                extra={
                    "request_id": request_id,
                    "status_code": exc.response.status_code,
                    "duration_ms": duration_ms,
                    "model": settings.LMSTUDIO_MODEL,
                    "error_body": error_body,
                },
            )
            raise ProviderError(
                f"LM Studio HTTP {exc.response.status_code} "
                f"(request_id={request_id}): {error_body}"
            ) from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(f"LM Studio request error: {exc}") from exc

        duration_ms = int((time.monotonic() - t_start) * 1000)

        try:
            body = response.json()
            choice = body["choices"][0]
            raw_content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
            usage = body.get("usage", {})
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                f"Unexpected LM Studio response structure (request_id={request_id}): {exc}"
            ) from exc

        logger.info(
            "LMStudio request completed",
            extra={
                "request_id": request_id,
                "duration_ms": duration_ms,
                "status_code": response.status_code,
                "finish_reason": finish_reason,
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
        )

        if finish_reason == "length":
            raise ProviderError(
                f"Model output was truncated (finish_reason=length). "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"max_tokens={settings.LMSTUDIO_MAX_TOKENS}. "
                f"Increase LMSTUDIO_MAX_TOKENS."
            )

        stripped = _strip_fences(raw_content)

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.error(
                "Model returned invalid JSON",
                extra={
                    "request_id": request_id,
                    "json_error": str(exc),
                    "output_chars": len(stripped),
                    "finish_reason": finish_reason,
                },
            )
            raise ProviderError(
                f"Model returned invalid JSON (request_id={request_id}): {exc}. "
                f"finish_reason={finish_reason}, output_chars={len(stripped)}"
            ) from exc

        try:
            return EditorialAnalysis(**data)
        except Exception as exc:
            raise ProviderError(
                f"AI output failed schema validation (request_id={request_id}): {exc}"
            ) from exc

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
            ) as client:
                r = await client.get(f"{self._settings.LMSTUDIO_BASE_URL}/models")
            return r.status_code == 200
        except Exception:
            return False

    def get_provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "lmstudio",
            "base_url": self._settings.LMSTUDIO_BASE_URL,
            "model": self._settings.LMSTUDIO_MODEL,
            "max_tokens": self._settings.LMSTUDIO_MAX_TOKENS,
            "temperature": self._settings.LMSTUDIO_TEMPERATURE,
            "timeout_seconds": self._settings.LMSTUDIO_TIMEOUT_SECONDS,
        }

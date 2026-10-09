import json
import time
import uuid
from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import ProviderError, ProviderUnavailableError
from app.core.logging import get_logger
from app.domain.models import EditorialAnalysis, EvidencePackage
from app.providers.base import LLMProvider

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a plain-language content advisor for a media team. "
    "You receive verified performance data about individual videos and social media posts. "
    "Your job is to explain what is happening with a piece of content and give clear, actionable advice.\n\n"
    "HOW TO WRITE:\n"
    "- Write like you are talking to a smart business owner or editor, not a data analyst.\n"
    "- Use short, direct sentences. Avoid bullet-point overload.\n"
    "- Good example: 'This video is performing strongly. The hook grabs attention quickly and the topic is hitting well with viewers right now. "
    "Consider making a follow-up video on the same theme while the momentum is there.'\n"
    "- Bad example: 'Observed signals indicate elevated performance requiring replication of content intent.'\n"
    "- Good example: 'This post is getting very little traction. The opening doesn't make it clear what viewers will learn. "
    "Try rewriting the first few seconds to lead with the most useful insight.'\n"
    "- Bad example: 'Performance degradation suggests optimization of content packaging and intent clarification.'\n\n"
    "WHAT TO INCLUDE:\n"
    "For HIGH_PERFORMING content:\n"
    "1. Say what is working, based only on the verified metrics and transcript if available.\n"
    "2. Give one clear action to extend or repeat this success.\n"
    "3. If the transcript shows why it works, mention it simply.\n"
    "For LOW_PERFORMING content:\n"
    "1. Say what the data shows is not working.\n"
    "2. Give one or two concrete things to try.\n"
    "3. Do not blame the creator — focus on what can be changed.\n\n"
    "STRICT RULES:\n"
    "- Never invent views, likes, dates, or rankings that are not in the data.\n"
    "- Never claim a cause unless the data or transcript actually supports it.\n"
    "- Do not use words like: velocity ratio, performance signal, content intent, "
    "elevated, BOOMING_SURGE, surge candidate, ML gating, baseline deviation.\n"
    "- Do not use ** markdown bold or bullet symbols like - or * in your output.\n"
    "- Write in plain paragraphs.\n"
    "- If you are not sure why something is performing a certain way, say so simply: "
    "'The exact reason is unclear from the available data, but here is what we can see.'\n"
    "- Do not publish timing recommendations unless actual historical data is provided.\n\n"
    "OUTPUT FORMAT:\n"
    "Return ONLY a valid JSON object with exactly these keys:\n"
    "content_intent (string — one plain sentence explaining what this content is about and why it performs this way),\n"
    "observed_signals (array of strings — plain sentences about what the numbers show, min 1),\n"
    "possible_contributing_factors (array of strings — plain sentences about likely causes, min 1),\n"
    "writer_recommendations (array of strings — plain actionable advice sentences, min 1),\n"
    "keyword_suggestions (array of strings),\n"
    "title_suggestions (array of strings — natural title rewrites),\n"
    "description_suggestions (array of strings — natural description rewrites),\n"
    "hashtag_suggestions (array of strings),\n"
    "cross_platform_ideas (array of strings — plain ideas for other platforms),\n"
    "confidence (float between 0.0 and 1.0),\n"
    "limitations (array of strings — plain notes on what we do not know),\n"
    "recommended_action (string — one plain sentence: the single most important thing to do next),\n"
    "publishing_timing (string — leave blank if no timing data is available).\n"
    "No markdown. No code fences. Output the JSON object and nothing else."
)


def _build_user_prompt(evidence: EvidencePackage) -> str:
    cand = getattr(evidence, "candidate", None)
    classification = cand.performance_level.upper() if cand else "HIGH_PERFORMING"

    title = cand.title if cand else evidence.content_metadata.title
    platform = cand.platform if cand else (evidence.content_metadata.platform or "unknown")
    content_type = cand.content_type if cand else (evidence.content_metadata.content_type or "Video")
    url = (cand.effective_url if cand else evidence.content_metadata.canonical_url) or "N/A"
    period = cand.period if cand else "30d"
    metric_name = cand.metric_name if cand else "views"
    curr_metric = cand.current_metric if cand else 0.0
    likes = cand.likes if cand else 0.0
    comments = cand.comments if cand else 0.0

    lines = [
        "=== VERIFIED CONTENT DATA — do not recalculate or contradict these figures ===",
        f"Title: {title}",
        f"Platform: {platform}",
        f"Content Type: {content_type}",
        f"Canonical URL: {url}",
        f"Analysis Period: {period}",
        f"Performance Classification: {classification}",
        f"Primary Metric ({metric_name}): {curr_metric:,.0f}",
        f"Likes: {likes:,.0f}",
        f"Comments: {comments:,.0f}",
    ]
    if cand and cand.published_at:
        lines.append(f"Published: {cand.published_at}")
    if cand and cand.peer_explanation:
        lines.append(f"Peer Comparison Evidence: {cand.peer_explanation}")

    if classification == "HIGH_PERFORMING":
        lines += [
            "",
            f"=== INTERPRETATION CONSTRAINT ({classification}) ===",
            "This content is performing well above average for the selected period.",
            "In plain language:",
            "1. Explain why it is performing strongly based strictly on verified metrics and transcript if available.",
            "2. Give one clear recommendation to extend or repeat this success.",
            "3. If the transcript is available, mention what the opening or topic tells us.",
            "4. Note anything we cannot be certain about from the data alone.",
        ]
    else:
        lines += [
            "",
            f"=== INTERPRETATION CONSTRAINT ({classification}) ===",
            "This content is underperforming relative to similar content in the selected period.",
            "In plain language:",
            "1. Identify observed limitations and bottlenecks based strictly on verified metrics.",
            "2. Give one or two concrete things the team could try.",
            "3. If the transcript is available, mention whether the opening or topic might be a factor.",
            "4. Note anything we cannot be certain about from the data alone.",
        ]

    if evidence.transcript_excerpt:
        lines += [
            "",
            "=== CONTENT TRANSCRIPT / TEXT (verified excerpt) ===",
            "Use this to understand the topic, opening, and structure. "
            "Do not claim the transcript proves any performance cause — only note what it shows.",
            evidence.transcript_excerpt,
        ]

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

    async def generate_structured_analysis(
        self,
        evidence: EvidencePackage,
        feedback: Optional[str] = None,
    ) -> EditorialAnalysis:
        settings = self._settings
        request_id = uuid.uuid4().hex[:12]
        user_prompt = _build_user_prompt(evidence)
        if feedback:
            user_prompt += (
                f"\n\n=== CORRECTION FEEDBACK (Previous output failed validation) ===\n"
                f"{feedback}\n"
                f"Please regenerate the complete JSON analysis correcting these issues. "
                f"Adhere strictly to verified evidence, use natural conversational tone, "
                f"avoid absolute certainty, and ensure all required JSON fields are present."
            )
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
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Cannot connect to LM Studio at {settings.LMSTUDIO_BASE_URL}. "
                f"Check: 1) Is LM Studio running on the target machine? "
                f"2) Is LMSTUDIO_BASE_URL in .env pointing to the correct IP address and port? "
                f"3) Is port 1234 accessible from this machine?"
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
            try:
                error_body = exc.response.text[:500]
            except Exception:
                error_body = "<unreadable>"
            status_code = exc.response.status_code
            logger.error(
                "LMStudio HTTP error",
                extra={
                    "request_id": request_id,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "model": settings.LMSTUDIO_MODEL,
                    "base_url": settings.LMSTUDIO_BASE_URL,
                    "error_body": error_body,
                },
            )
            # HTTP 4xx from LM Studio that indicates server-side unavailability
            # (e.g. "No models loaded") is an availability issue, not a model
            # output validation failure.  Raise ProviderUnavailableError so the
            # caller (bulk_scanner) records llm_status = "unavailable" and the
            # frontend renders "LLM Unavailable" rather than "Validation Failed".
            no_model_indicators = [
                "no models loaded",
                "no model loaded",
                "please load a model",
                "lms load",
                "model not found",
                "model_not_found",
            ]
            error_lower = error_body.lower()
            if status_code in (400, 404, 503) and any(ind in error_lower for ind in no_model_indicators):
                raise ProviderUnavailableError(
                    f"LM Studio has no model loaded or the configured model is not found "
                    f"(request_id={request_id}, HTTP {status_code}, model={settings.LMSTUDIO_MODEL}, "
                    f"url={settings.LMSTUDIO_BASE_URL}). "
                    f"Check: 1) Is the correct model loaded in LM Studio? "
                    f"2) Does LMSTUDIO_MODEL in .env match the loaded model identifier? "
                    f"3) Is LMSTUDIO_BASE_URL pointing to the correct IP address and port?"
                ) from exc
            if status_code >= 500:
                raise ProviderUnavailableError(
                    f"LM Studio server error HTTP {status_code} "
                    f"(request_id={request_id}, url={settings.LMSTUDIO_BASE_URL}): {error_body}"
                ) from exc
            raise ProviderError(
                f"LM Studio HTTP {status_code} "
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

        try:
            from app.providers.utils import extract_and_normalize_analysis
            return extract_and_normalize_analysis(raw_content, evidence, request_id)
        except Exception as exc:
            logger.error(
                "Model returned invalid output or failed schema",
                extra={
                    "request_id": request_id,
                    "error": str(exc),
                    "output_chars": len(raw_content),
                    "finish_reason": finish_reason,
                },
            )
            raise ProviderError(
                f"AI output failed schema/extraction (request_id={request_id}): {exc}"
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

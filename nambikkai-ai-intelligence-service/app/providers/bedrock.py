import asyncio
import json
import time
import uuid
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from app.core.config import get_settings
from app.core.exceptions import ProviderError, ProviderUnavailableError
from app.core.logging import get_logger
from app.domain.models import EditorialAnalysis, EvidencePackage
from app.providers.base import LLMProvider
from app.providers.lmstudio import _SYSTEM_PROMPT, _build_user_prompt, _strip_fences

logger = get_logger(__name__)


class BedrockProvider(LLMProvider):
    """
    AWS Bedrock production provider.

    Authentication relies entirely on the boto3 credential chain:
    environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
    AWS_SESSION_TOKEN), ~/.aws/credentials profile, or an attached IAM role.
    No credentials are hardcoded or logged.

    Activate with: AI_PROVIDER=bedrock
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        # Clients are initialised eagerly to avoid lazy-init races and to
        # surface credential errors at startup rather than at first request.
        self._runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=self._settings.AWS_REGION,
        )
        self._bedrock_client = boto3.client(
            "bedrock",
            region_name=self._settings.AWS_REGION,
        )

    async def generate_structured_analysis(self, evidence: EvidencePackage) -> EditorialAnalysis:
        settings = self._settings
        request_id = uuid.uuid4().hex[:12]
        user_prompt = _build_user_prompt(evidence)
        input_chars = len(_SYSTEM_PROMPT) + len(user_prompt)

        # Bedrock Converse API — unified interface across all supported models.
        # Uses the same logical prompt contract as LMStudioProvider.
        payload = {
            "modelId": settings.BEDROCK_MODEL_ID,
            "system": [{"text": _SYSTEM_PROMPT}],
            "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
            "inferenceConfig": {
                "maxTokens": settings.BEDROCK_MAX_TOKENS,
                "temperature": settings.BEDROCK_TEMPERATURE,
            },
        }

        logger.info(
            "Bedrock request starting",
            extra={
                "request_id": request_id,
                "provider": "bedrock",
                "model_id": settings.BEDROCK_MODEL_ID,
                "region": settings.AWS_REGION,
                "input_chars": input_chars,
            },
        )

        t_start = time.monotonic()
        try:
            # boto3 is synchronous. Run it in a thread pool so the asyncio
            # event loop is not blocked during the Bedrock network call.
            response = await asyncio.to_thread(
                self._runtime_client.converse, **payload
            )
        except NoCredentialsError as exc:
            raise ProviderUnavailableError(
                "AWS credentials not found. Configure via environment variables, "
                "~/.aws/credentials, or an IAM role."
            ) from exc
        except ClientError as exc:
            duration_ms = int((time.monotonic() - t_start) * 1000)
            code = exc.response["Error"]["Code"]
            # Do not log the full ClientError — it may contain account details.
            logger.error(
                "Bedrock ClientError",
                extra={"request_id": request_id, "error_code": code, "duration_ms": duration_ms},
            )
            if code in ("AccessDeniedException", "UnauthorizedException"):
                raise ProviderUnavailableError(
                    f"Bedrock access denied (request_id={request_id}). "
                    "Check IAM permissions for bedrock:InvokeModel."
                ) from exc
            if code == "ThrottlingException":
                raise ProviderUnavailableError(
                    f"Bedrock throttled (request_id={request_id}). Retry later."
                ) from exc
            if code == "ModelTimeoutException":
                raise ProviderUnavailableError(
                    f"Bedrock model timed out (request_id={request_id})."
                ) from exc
            raise ProviderError(
                f"Bedrock ClientError {code} (request_id={request_id})"
            ) from exc
        except BotoCoreError as exc:
            raise ProviderUnavailableError(
                f"Bedrock connectivity error (request_id={request_id}): {type(exc).__name__}"
            ) from exc

        duration_ms = int((time.monotonic() - t_start) * 1000)

        # ── Parse Converse response ────────────────────────────────────────
        # Converse response shape:
        # {"output": {"message": {"content": [{"text": "..."}]}},
        #  "stopReason": "end_turn" | "max_tokens" | ...,
        #  "usage": {"inputTokens": N, "outputTokens": N}}
        try:
            stop_reason = response.get("stopReason")
            usage = response.get("usage", {})
            raw_content = response["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                f"Unexpected Bedrock response structure (request_id={request_id}): {exc}"
            ) from exc

        logger.info(
            "Bedrock request completed",
            extra={
                "request_id": request_id,
                "duration_ms": duration_ms,
                "stop_reason": stop_reason,
                "input_tokens": usage.get("inputTokens"),
                "output_tokens": usage.get("outputTokens"),
            },
        )

        if stop_reason == "max_tokens":
            raise ProviderError(
                f"Model output was truncated (stopReason=max_tokens). "
                f"output_tokens={usage.get('outputTokens')}, "
                f"max_tokens={settings.BEDROCK_MAX_TOKENS}. "
                f"Increase BEDROCK_MAX_TOKENS."
            )

        stripped = _strip_fences(raw_content)

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.error(
                "Bedrock model returned invalid JSON",
                extra={
                    "request_id": request_id,
                    "json_error": str(exc),
                    "output_chars": len(stripped),
                    "stop_reason": stop_reason,
                },
            )
            raise ProviderError(
                f"Bedrock model returned invalid JSON (request_id={request_id}): {exc}. "
                f"stop_reason={stop_reason}, output_chars={len(stripped)}"
            ) from exc

        try:
            return EditorialAnalysis(**data)
        except Exception as exc:
            raise ProviderError(
                f"Bedrock AI output failed schema validation (request_id={request_id}): {exc}"
            ) from exc

    async def health_check(self) -> bool:
        """
        Lightweight check: list accessible Bedrock foundation models.
        Uses the cached bedrock (not bedrock-runtime) client — no inference cost.
        Returns True only if the call succeeds without credential/access errors.
        """
        try:
            await asyncio.to_thread(
                self._bedrock_client.list_foundation_models,
                byOutputModality="TEXT",
            )
            return True
        except (NoCredentialsError, ClientError, BotoCoreError):
            return False
        except Exception:
            return False

    def get_provider_metadata(self) -> dict[str, Any]:
        return {
            "provider": "bedrock",
            "region": self._settings.AWS_REGION,
            "model_id": self._settings.BEDROCK_MODEL_ID,
            "status": "active",
        }

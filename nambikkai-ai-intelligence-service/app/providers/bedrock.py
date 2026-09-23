import asyncio
import json
import os
import time
import uuid
from typing import Any

import boto3
from botocore.config import Config
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

    Authentication supports explicit environment/Pydantic settings as well
    as the standard boto3 credential chain (~/.aws/credentials, IAM roles).
    No credentials are hardcoded, logged, or exposed in exceptions/metadata.

    Activate with: AI_PROVIDER=bedrock
    """

    def __init__(self) -> None:
        self._settings = get_settings()

        session_kwargs: dict[str, Any] = {}
        if self._settings.AWS_ACCESS_KEY_ID:
            session_kwargs["aws_access_key_id"] = self._settings.AWS_ACCESS_KEY_ID
        if self._settings.AWS_SECRET_ACCESS_KEY:
            session_kwargs["aws_secret_access_key"] = self._settings.AWS_SECRET_ACCESS_KEY
        if self._settings.AWS_SESSION_TOKEN:
            session_kwargs["aws_session_token"] = self._settings.AWS_SESSION_TOKEN

        boto_config = Config(
            connect_timeout=10,
            read_timeout=float(self._settings.BEDROCK_TIMEOUT_SECONDS),
            retries={"max_attempts": 2},
        )

        session = boto3.Session(**session_kwargs)
        self._runtime_client = session.client(
            "bedrock-runtime",
            region_name=self._settings.AWS_REGION,
            config=boto_config,
        )
        self._bedrock_client = session.client(
            "bedrock",
            region_name=self._settings.AWS_REGION,
            config=boto_config,
        )

    def _get_effective_model_id(self) -> str:
        model_id = self._settings.BEDROCK_MODEL_ID
        # Nova models in ap-south-1 (and other non-primary regions) require the
        # global. cross-region inference prefix on the Converse API.
        # Apply it automatically when the configured ID is a bare amazon.nova-* ID
        # so the operator doesn't have to remember the prefix.
        if model_id and not model_id.startswith("global.") and model_id.startswith("amazon.nova-"):
            return f"global.{model_id}"
        return model_id

    async def generate_structured_analysis(self, evidence: EvidencePackage) -> EditorialAnalysis:
        settings = self._settings
        request_id = uuid.uuid4().hex[:12]
        user_prompt = _build_user_prompt(evidence)
        input_chars = len(_SYSTEM_PROMPT) + len(user_prompt)
        effective_model_id = self._get_effective_model_id()

        payload = {
            "modelId": effective_model_id,
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
                "model_id": effective_model_id,
                "region": settings.AWS_REGION,
                "timeout_seconds": settings.BEDROCK_TIMEOUT_SECONDS,
                "max_tokens": settings.BEDROCK_MAX_TOKENS,
                "temperature": settings.BEDROCK_TEMPERATURE,
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
                "total_tokens": usage.get("totalTokens"),
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
        Uses the cached bedrock (not bedrock-runtime) client — zero inference cost.
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
        has_creds = bool(
            self._settings.AWS_ACCESS_KEY_ID
            or os.environ.get("AWS_ACCESS_KEY_ID")
        )
        return {
            "provider": "bedrock",
            "region": self._settings.AWS_REGION,
            "model_id": self._get_effective_model_id(),
            "max_tokens": self._settings.BEDROCK_MAX_TOKENS,
            "temperature": self._settings.BEDROCK_TEMPERATURE,
            "timeout_seconds": self._settings.BEDROCK_TIMEOUT_SECONDS,
            "credentials_configured": has_creds,
        }

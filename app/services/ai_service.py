import time
import uuid
import json
from typing import Optional, Generator, Callable
from dataclasses import dataclass, field, asdict

from app.llm.router import router as llm_router
from app.config.settings import settings
from app.models.ai_audit_log import save_log
from app.utils.logger import logger


@dataclass
class AIResponse:
    content: str
    request_id: str
    feature_name: str
    model: str
    latency_ms: int
    ai_used: bool
    fallback: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "request_id": self.request_id,
            "feature_name": self.feature_name,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "ai_used": self.ai_used,
            "fallback": self.fallback,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class AIService:
    """Central AI Service — all AI features MUST use this.

    Single source of truth for AI audit logging. Every call is logged
    to the ai_audit_logs table with feature_name, latency, tokens, etc.
    """

    def generate(
        self,
        prompt: str,
        feature_name: str,
        system: Optional[str] = None,
        endpoint: Optional[str] = None,
        user_id: str = "anonymous",
        fallback_fn: Optional[Callable[[], str]] = None,
        **kwargs,
    ) -> AIResponse:
        request_id = uuid.uuid4().hex[:12].upper()
        model = settings.OLLAMA_MODEL
        start = time.time()
        fallback_used = False
        content = ""
        status = "SUCCESS"
        error_message = None

        try:
            content = llm_router.generate(prompt=prompt, system=system, **kwargs)
        except Exception as e:
            logger.error(f"[AIService] {feature_name} LLM failed: {e}")
            if fallback_fn:
                logger.info(f"[AIService] {feature_name} using fallback")
                fallback_used = True
                try:
                    content = fallback_fn()
                except Exception as fe:
                    logger.error(f"[AIService] {feature_name} fallback also failed: {fe}")
                    content = ""
                    status = "FAILED"
                    error_message = str(fe)
            else:
                status = "FAILED"
                error_message = str(e)

        end = time.time()
        latency_ms = int((end - start) * 1000)
        prompt_tokens = len(prompt.split()) + (len((system or "").split()) if system else 0)
        completion_tokens = len(content.split()) if content else 0

        try:
            save_log(
                request_id=request_id,
                feature_name=feature_name,
                endpoint=endpoint or "",
                model=model,
                user_id=user_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                status=status,
                fallback_used=fallback_used,
                error_message=error_message,
                prompt_preview=prompt[:300],
                response_preview=content[:300] if content else "",
            )
        except Exception as e:
            logger.error(f"[AIService] Audit log save failed: {e}")

        logger.info(
            f"\n{'='*50}\n"
            f"  Feature      : {feature_name}\n"
            f"  Model        : {model}\n"
            f"  Endpoint     : {endpoint or 'N/A'}\n"
            f"  Request ID   : {request_id}\n"
            f"  Prompt Tokens: {prompt_tokens}\n"
            f"  Comp Tokens  : {completion_tokens}\n"
            f"  Latency      : {latency_ms} ms\n"
            f"  Status       : {status}\n"
            f"  Fallback     : {'YES' if fallback_used else 'NO'}\n"
            f"{'='*50}"
        )

        return AIResponse(
            content=content,
            request_id=request_id,
            feature_name=feature_name,
            model=model,
            latency_ms=latency_ms,
            ai_used=not fallback_used,
            fallback=fallback_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


ai_service = AIService()

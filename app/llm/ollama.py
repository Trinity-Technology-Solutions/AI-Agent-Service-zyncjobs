import time
from typing import Optional
import ollama
from ollama import Client

from app.config.settings import settings
from app.utils.logger import logger
from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """Pure LLM client — no audit logging. Audit is handled by AIService wrapper."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.OLLAMA_MODEL
        self.client = Client(
            host=settings.OLLAMA_BASE_URL,
            timeout=settings.OLLAMA_TIMEOUT,
        )

    def _build_options(self, kwargs: dict) -> dict:
        options = {**settings.OLLAMA_OPTIONS, **kwargs.pop("options", {})}
        if settings.OLLAMA_STOP:
            options["stop"] = settings.OLLAMA_STOP
        options.update(kwargs.pop("stop_overrides", {}))
        return options

    def _call_with_retry(self, call_fn, **call_kwargs):
        last_exc = None
        for attempt in range(1, settings.OLLAMA_RETRY_MAX + 1):
            try:
                return call_fn(**call_kwargs)
            except Exception as e:
                last_exc = e
                logger.warn(f"OllamaLLM attempt={attempt}/{settings.OLLAMA_RETRY_MAX} failed", error=str(e))
                if attempt < settings.OLLAMA_RETRY_MAX:
                    time.sleep(settings.OLLAMA_RETRY_DELAY * attempt)
        raise last_exc

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = self._build_options(kwargs)
        response = self._call_with_retry(
            self.client.chat,
            model=self.model,
            messages=messages,
            options=options,
            **kwargs,
        )
        return response["message"]["content"]

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs,
    ):
        import threading

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = self._build_options(kwargs)
        result_queue: list[str] = []
        exception: list[Exception] = []

        def _do_stream():
            try:
                stream = self._call_with_retry(
                    self.client.chat,
                    model=self.model,
                    messages=messages,
                    stream=True,
                    options=options,
                    **kwargs,
                )
                for chunk in stream:
                    result_queue.append(chunk["message"]["content"])
                result_queue.append(None)
            except Exception as e:
                exception.append(e)
                result_queue.append(None)

        t = threading.Thread(target=_do_stream, daemon=True)
        t.start()

        idx = 0
        while True:
            while idx < len(result_queue):
                val = result_queue[idx]
                idx += 1
                if val is None:
                    if exception:
                        raise exception[0]
                    return
                yield val
            time.sleep(0.01)

"""Shared Ollama service for all brains."""
import httpx
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from recruitment_ai.config.settings import settings


class OllamaService:
    """Async Ollama client with model routing."""

    # Dev fallback — used when a brain key is not in settings.OLLAMA_MODELS
    _DEV_MODEL = "qwen2.5:3b"

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.timeout = settings.OLLAMA_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def get_model(self, brain_name: str) -> str:
        """Get model for a brain — reads from settings.OLLAMA_MODELS, falls back to dev model."""
        return settings.OLLAMA_MODELS.get(brain_name.lower(), self._DEV_MODEL)

    async def _do_generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        client = await self._get_client()
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": 0.9},
        }
        response = await client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    async def generate(
        self,
        brain_name: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Generate response using Ollama with retry."""
        model = self.get_model(brain_name)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exception = None
        for attempt in range(3):
            try:
                return await self._do_generate(model, messages, temperature, max_tokens)
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                import asyncio
                await asyncio.sleep(2 ** attempt)
        raise last_exception or Exception("Ollama generate failed after 3 retries")

    async def embed(self, text: str, model: str = "nomic-embed-text") -> list[float]:
        """Generate embeddings."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        response.raise_for_status()
        return response.json().get("embedding", [])

    async def health_check(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


ollama_service = OllamaService()
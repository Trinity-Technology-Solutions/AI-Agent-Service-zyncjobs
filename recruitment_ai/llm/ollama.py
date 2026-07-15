"""Ollama LLM provider — used in development."""
import httpx
from typing import Optional
from recruitment_ai.config.settings import settings


class OllamaProvider:

    _DEV_MODEL = "qwen2.5:3b"

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.timeout = settings.OLLAMA_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _get_model(self, brain_name: str) -> str:
        return settings.OLLAMA_MODELS.get(brain_name.lower(), self._DEV_MODEL)

    async def generate(
        self,
        brain_name: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        model = self._get_model(brain_name)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        last_exc = None
        for attempt in range(3):
            try:
                client = await self._get_client()
                resp = await client.post(f"{self.base_url}/api/chat", json={"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": 0.9}})
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "")
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_exc = e
                import asyncio
                await asyncio.sleep(2 ** attempt)
        raise last_exc or Exception("Ollama generate failed after 3 retries")

    async def embed(self, text: str) -> list[float]:
        client = await self._get_client()
        resp = await client.post(f"{self.base_url}/api/embeddings", json={"model": "nomic-embed-text", "prompt": text})
        resp.raise_for_status()
        return resp.json().get("embedding", [])

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            resp = await client.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

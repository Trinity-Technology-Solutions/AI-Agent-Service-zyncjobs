"""Shared Ollama service for all brains."""
import httpx
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from recruitment_ai.config.settings import settings


class OllamaService:
    """Async Ollama client with model routing."""

    MODEL_MAP = {
        "chatbot": "qwen2.5:3b",
        "job_parser": "qwen2.5:3b",
        "jd_generator": "qwen2.5:3b",
        "resume_parser": "qwen2.5:3b",
        "ats_scanner": "qwen2.5:3b",
        "job_matching": "qwen2.5:3b",
        "career_advice": "qwen2.5:3b",
        "recruiter": "qwen2.5:3b",
        "skill_assessment": "qwen2.5:3b",
        "interview_prep": "qwen2.5:3b",
        "resume_builder": "qwen2.5:3b",
    }

    # Production model map (uncomment when RAM available):
    # MODEL_MAP = {
    #     "chatbot": "llama3.1:8b",
    #     "job_parser": "qwen3:8b",
    #     "jd_generator": "llama3.1:8b",
    #     "resume_parser": "qwen3:8b",
    #     "ats_scanner": "qwen3:8b",
    #     "job_matching": "qwen3:8b",
    #     "recruiter": "llama3.1:8b",
    #     "career_advice": "llama3.1:8b",
    #     "skill_assessment": "qwen3:8b",
    #     "interview_prep": "llama3.1:8b",
    #     "resume_builder": "llama3.1:8b",
    # }

    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.timeout = settings.OLLAMA_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def get_model(self, brain_name: str) -> str:
        """Get model for a brain."""
        return self.MODEL_MAP.get(brain_name.lower(), settings.OLLAMA_MODEL)

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
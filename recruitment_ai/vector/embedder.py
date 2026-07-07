import numpy as np
from typing import Optional
import httpx
from recruitment_ai.config.settings import settings


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self._client = httpx.Client(timeout=30)

    def embed(self, text: str) -> list[float]:
        try:
            res = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            res.raise_for_status()
            return res.json()["embedding"]
        except Exception:
            return [0.0] * 768

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(np.dot(a, b) / norm)

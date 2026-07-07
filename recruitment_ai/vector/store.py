"""Vector store abstraction layer - Qdrant with ChromaDB fallback."""
from typing import Optional, Any
from pathlib import Path
from recruitment_ai.config.settings import settings
from recruitment_ai.shared.ollama_service import ollama_service
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VectorDoc:
    text: str
    metadata: dict
    score: float = 0.0


class VectorStore:
    """Abstracts vector storage. Uses Qdrant when available, falls back to existing ChromaDB."""

    def __init__(self, collection_name: str = settings.QDRANT_COLLECTION):
        self.collection_name = collection_name
        self._qdrant_client: Optional[Any] = None
        self._chroma_ingester: Optional[Any] = None

    @property
    def chroma_ingester(self):
        if self._chroma_ingester is None:
            from recruitment_ai.vector.ingest import ingester
            self._chroma_ingester = ingester
        return self._chroma_ingester

    async def search(self, query: str, top_k: int = 5) -> list[VectorDoc]:
        """Search using ChromaDB (local fallback until Qdrant is available)."""
        try:
            docs = self.chroma_ingester.search(query, top_k=top_k)
            from recruitment_ai.vector.reranker import rerank
            docs = rerank(query, docs, top_k=min(3, top_k))

            return [
                VectorDoc(
                    text=d.text,
                    metadata={
                        "url": d.metadata.get("url", ""),
                        "title": d.metadata.get("title", "ZyncJobs"),
                        "category": d.metadata.get("category", ""),
                    },
                    score=d.score,
                )
                for d in docs
                if d.score < 0.45  # Only return highly relevant docs (cosine distance < 0.45)
            ]
        except Exception as e:
            logger.warning(f"ChromaDB search failed: {e}")
            return []

    async def embed(self, text: str) -> list[float]:
        """Embed text using Ollama embedding model."""
        return await ollama_service.embed(model="nomic-embed-text", text=text)

    @property
    def count(self) -> int:
        try:
            return self.chroma_ingester.count
        except Exception:
            return 0


vector_store = VectorStore()
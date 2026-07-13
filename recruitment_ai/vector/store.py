"""Vector store abstraction — Qdrant primary, ChromaDB fallback.
Architecture doc: RAG = Qdrant + BGE-M3 embeddings.
Qdrant is connected on startup; if unavailable ChromaDB is used silently.
"""
from typing import Optional, Any
from dataclasses import dataclass
import logging
from recruitment_ai.config.settings import settings
from recruitment_ai.shared.ollama_service import ollama_service

logger = logging.getLogger(__name__)

EMBED_MODEL = "bge-m3"          # architecture doc: BGE-M3 embeddings
EMBED_FALLBACK = "nomic-embed-text"
SCORE_THRESHOLD = 0.45          # cosine distance — lower = more relevant


@dataclass
class VectorDoc:
    text: str
    metadata: dict
    score: float = 0.0


class VectorStore:
    """Qdrant primary, ChromaDB fallback. Same interface regardless of backend."""

    def __init__(self, collection_name: str = settings.QDRANT_COLLECTION):
        self.collection_name = collection_name
        self._qdrant: Optional[Any] = None          # qdrant_client.AsyncQdrantClient
        self._qdrant_available = False
        self._embed_model = EMBED_MODEL

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Try to connect to Qdrant. Falls back to ChromaDB silently."""
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._qdrant = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=5,
            )
            # Verify connection
            await self._qdrant.get_collections()

            # Ensure collection exists
            existing = [c.name for c in (await self._qdrant.get_collections()).collections]
            if self.collection_name not in existing:
                await self._qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
                logger.info("Qdrant collection created: %s", self.collection_name)

            # Verify BGE-M3 available, fall back to nomic-embed-text
            self._embed_model = await self._resolve_embed_model()
            self._qdrant_available = True
            logger.info("Qdrant connected: %s:%s (embed=%s)", settings.QDRANT_HOST, settings.QDRANT_PORT, self._embed_model)

        except Exception as e:
            self._qdrant_available = False
            self._qdrant = None
            logger.warning("Qdrant unavailable — using ChromaDB fallback: %s", e)

    async def _resolve_embed_model(self) -> str:
        """Use BGE-M3 if Ollama has it, else nomic-embed-text."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3) as c:
                res = await c.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                models = [m["name"] for m in res.json().get("models", [])]
                if any("bge-m3" in m for m in models):
                    return EMBED_MODEL
        except Exception:
            pass
        return EMBED_FALLBACK

    async def close(self) -> None:
        if self._qdrant:
            await self._qdrant.close()
            self._qdrant = None
            self._qdrant_available = False

    # ── Embed ──────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float]:
        """Embed using BGE-M3 (Qdrant path) or nomic-embed-text (ChromaDB path)."""
        model = self._embed_model if self._qdrant_available else EMBED_FALLBACK
        return await ollama_service.embed(text=text, model=model)

    # ── Search ─────────────────────────────────────────────────────────────

    async def search(self, query: str, top_k: int = 5) -> list[VectorDoc]:
        """Search Qdrant if available, else ChromaDB."""
        if self._qdrant_available:
            return await self._search_qdrant(query, top_k)
        return await self._search_chroma(query, top_k)

    async def _search_qdrant(self, query: str, top_k: int) -> list[VectorDoc]:
        try:
            vector = await self.embed(query)
            hits = await self._qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                score_threshold=1.0 - SCORE_THRESHOLD,  # Qdrant uses similarity (higher=better)
            )
            docs = [
                VectorDoc(
                    text=h.payload.get("text", ""),
                    metadata={
                        "url": h.payload.get("url", ""),
                        "title": h.payload.get("title", "ZyncJobs"),
                        "category": h.payload.get("category", ""),
                    },
                    score=1.0 - h.score,  # convert similarity → distance for consistent interface
                )
                for h in hits
            ]
            from recruitment_ai.vector.reranker import rerank
            return rerank(query, docs, top_k=min(3, top_k))
        except Exception as e:
            logger.warning("Qdrant search failed, falling back to ChromaDB: %s", e)
            return await self._search_chroma(query, top_k)

    async def _search_chroma(self, query: str, top_k: int) -> list[VectorDoc]:
        try:
            from recruitment_ai.vector.ingest import ingester
            from recruitment_ai.vector.reranker import rerank
            docs = ingester.search(query, top_k=top_k)
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
                if d.score < SCORE_THRESHOLD
            ]
        except Exception as e:
            logger.warning("ChromaDB search failed: %s", e)
            return []

    # ── Upsert ─────────────────────────────────────────────────────────────

    async def upsert(self, doc_id: str, text: str, metadata: dict) -> None:
        """Insert or update a document in Qdrant."""
        if not self._qdrant_available:
            return
        try:
            from qdrant_client.models import PointStruct
            vector = await self.embed(text)
            await self._qdrant.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=doc_id, vector=vector, payload={"text": text, **metadata})],
            )
        except Exception as e:
            logger.warning("Qdrant upsert failed: %s", e)

    # ── Stats ──────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        if self._qdrant_available:
            return 0  # async count not available as sync property — use /knowledge/stats endpoint
        try:
            from recruitment_ai.vector.ingest import ingester
            return ingester.count
        except Exception:
            return 0

    @property
    def backend(self) -> str:
        return "qdrant" if self._qdrant_available else "chromadb"


vector_store = VectorStore()
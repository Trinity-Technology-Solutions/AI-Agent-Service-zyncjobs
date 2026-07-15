"""Vector store package — Qdrant primary, ChromaDB fallback, BGE-M3 embeddings."""
from recruitment_ai.vector.base import Document
from recruitment_ai.vector.store import VectorStore, VectorDoc, vector_store
from recruitment_ai.vector.embedder import OllamaEmbedder
from recruitment_ai.vector.reranker import rerank
from recruitment_ai.vector.ingester import ingest_knowledge, ingest_file, reindex_all

__all__ = [
    "Document", "VectorStore", "VectorDoc", "vector_store",
    "OllamaEmbedder", "rerank",
    "ingest_knowledge", "ingest_file", "reindex_all",
]

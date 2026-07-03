from typing import Optional
from .base import Document
from .vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.store = vector_store

    def retrieve(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[Document]:
        results = self.store.search(query, top_k=top_k)
        return [r for r in results if r.score >= min_score]

    def retrieve_by_source(self, query: str, source: str, top_k: int = 3) -> list[Document]:
        all_results = self.store.search(query, top_k=top_k * 3)
        filtered = [r for r in all_results if r.source == source]
        return filtered[:top_k]

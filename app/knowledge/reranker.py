from typing import Optional
from app.knowledge.base import Document


def rerank(query: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    """Score-based reranking using cosine distance from ChromaDB."""
    if not docs:
        return []

    scored = [(d, 1.0 - d.score) for d in docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    reranked = [d for d, _ in scored[:top_k]]
    return reranked

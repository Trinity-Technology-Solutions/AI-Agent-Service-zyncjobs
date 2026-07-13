"""Simple keyword-based reranker for vector search results.
Architecture doc: RAG → Qdrant + BGE-M3 + reranking.
Reranks retrieved docs by keyword overlap with query — no extra model needed.
"""
import re
from recruitment_ai.vector.base import Document


def rerank(query: str, docs: list, top_k: int = 3) -> list:
    """Rerank docs by keyword overlap with query. Works with VectorDoc and Document."""
    if not docs:
        return []

    query_tokens = set(re.findall(r"\w+", query.lower()))

    def score(doc) -> float:
        text = getattr(doc, "text", "") or ""
        doc_tokens = set(re.findall(r"\w+", text.lower()))
        overlap = len(query_tokens & doc_tokens)
        base_score = getattr(doc, "score", 0.5)
        # Lower score = more relevant (distance), so subtract overlap bonus
        return base_score - (overlap * 0.01)

    return sorted(docs, key=score)[:top_k]

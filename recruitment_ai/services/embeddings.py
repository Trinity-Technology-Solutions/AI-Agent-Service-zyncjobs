"""Embedding service — generates vector embeddings for text.
Matches roadmap spec: services/embeddings.py
"""
from recruitment_ai.llm import llm_router


async def embed_text(text: str) -> list[float]:
    return await llm_router.embed(text)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    results = []
    for t in texts:
        results.append(await llm_router.embed(t))
    return results

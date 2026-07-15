"""Retrieve context node — fetches relevant docs from vector store (Qdrant/ChromaDB).
Populates BrainState.retrieved_context + backward-compat context["rag_context"].
Only runs for intents that benefit from RAG. Skipped on cache hits.
"""
import logging
from recruitment_ai.brains.base import BrainState

logger = logging.getLogger(__name__)

_RAG_INTENTS = {
    "CHAT", "CAREER_ADVICE", "CAREER_ROADMAP", "SKILL_GAP",
    "JOB_MATCH", "ATS_SCORE", "INTERVIEW_PREP", "RESUME_BUILDER",
}


async def retrieve_context_node(state: BrainState) -> BrainState:
    query = (state.query or "").strip()
    intent = state.intent or "CHAT"

    if not query or intent not in _RAG_INTENTS:
        return state
    if state.metadata.get("cache_hit"):
        return state

    try:
        from recruitment_ai.vector.store import vector_store
        docs = await vector_store.search(query, top_k=5)
        if docs:
            rag = [
                {
                    "text": d.text,
                    "title": d.metadata.get("title", ""),
                    "url": d.metadata.get("url", ""),
                    "score": round(d.score, 4),
                }
                for d in docs
            ]
            state.retrieved_context = rag
            state.context["rag_context"] = rag
            state.metadata["rag_docs"] = len(rag)
            logger.debug("RAG: %d docs retrieved for intent %s", len(rag), intent)
    except Exception as e:
        logger.warning("RAG retrieval failed: %s", e)

    return state

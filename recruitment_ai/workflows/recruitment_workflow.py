"""LangGraph workflow for the recruitment AI platform."""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from recruitment_ai.shared.brain import BrainState
from recruitment_ai.shared.master_brain import master_brain
from recruitment_ai.shared.intent_classifier import intent_classifier
from recruitment_ai.services.cache_service import cache_service
from recruitment_ai.services.memory_service import memory_service, CHAT_INTENTS


class WorkflowState(TypedDict):
    query: Optional[str]
    intent: Optional[str]
    user_id: Optional[str]
    session_id: Optional[str]
    user_role: Optional[str]
    file_content: Optional[str]
    file_type: Optional[str]
    context: Optional[dict]
    result: Optional[dict]
    error: Optional[str]
    metadata: dict


async def authenticate_node(state: WorkflowState) -> WorkflowState:
    """Authenticate the request."""
    if not state.get("user_id"):
        state["error"] = "Authentication required"
    return state


async def load_memory_node(state: WorkflowState) -> WorkflowState:
    """Load session history and inject into context so brains can use it."""
    session_id = state.get("session_id")
    history = await memory_service.load(session_id)
    if history:
        ctx = dict(state.get("context") or {})
        ctx["history"] = history
        state["context"] = ctx
        state["metadata"]["history_turns"] = len(history)
    return state


# Intents that benefit from RAG context injection
_RAG_INTENTS = {
    "CHAT", "CAREER_ADVICE", "CAREER_ROADMAP", "SKILL_GAP",
    "JOB_MATCH", "ATS_SCORE", "INTERVIEW_PREP",
    "RESUME_BUILDER",
}


async def retrieve_context_node(state: WorkflowState) -> WorkflowState:
    """Retrieve relevant context from vector store and inject into state.context."""
    query = state.get("query") or ""
    intent = state.get("intent") or "CHAT"

    # Skip RAG for file-based tasks and cache hits
    if not query.strip() or intent not in _RAG_INTENTS:
        return state
    if state.get("metadata", {}).get("cache_hit"):
        return state

    try:
        from recruitment_ai.vector.store import vector_store
        docs = await vector_store.search(query, top_k=5)
        if docs:
            ctx = dict(state.get("context") or {})
            ctx["rag_context"] = [
                {
                    "text": d.text,
                    "title": d.metadata.get("title", ""),
                    "url": d.metadata.get("url", ""),
                    "score": round(d.score, 4),
                }
                for d in docs
            ]
            state["context"] = ctx
            state["metadata"]["rag_docs"] = len(docs)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("RAG retrieval failed: %s", e)

    return state


async def intent_detection_node(state: WorkflowState) -> WorkflowState:
    """Detect intent and route to appropriate brain."""
    brain_state = BrainState(**state)
    brain_state = await intent_classifier.classify(brain_state)
    state["intent"] = brain_state.intent
    return state


async def planner_node(state: WorkflowState) -> WorkflowState:
    """Plan which brain to execute — check Redis cache after intent is known."""
    intent = state.get("intent", "CHAT")
    state["metadata"]["planned_brain"] = intent
    query = state.get("query") or ""
    if query and intent:
        cached = await cache_service.get(intent, query)
        if cached:
            state["result"] = cached
            state["metadata"]["cache_hit"] = True
    return state


async def execute_brain_node(state: WorkflowState) -> WorkflowState:
    """Execute the selected brain — skipped if cache already populated result."""
    if state.get("metadata", {}).get("cache_hit"):
        return state

    brain_state = BrainState(**state)
    intent = brain_state.intent or "CHAT"
    brain = master_brain.brains.get(intent)

    if not brain:
        state["error"] = f"No brain found for intent: {intent}"
        return state

    if not await brain.validate_input(brain_state):
        state["error"] = f"Invalid input for {brain.name}"
        return state

    try:
        brain_state = await brain.run(brain_state)
        brain_state = await brain.post_process(brain_state)
    except Exception as e:
        brain_state = await brain.handle_error(brain_state, e)

    state.update(brain_state.model_dump())
    return state


async def validate_node(state: WorkflowState) -> WorkflowState:
    """Validate the result."""
    if state.get("error"):
        state["result"] = {"error": state["error"]}
    return state


async def store_memory_node(state: WorkflowState) -> WorkflowState:
    """Persist result to Redis cache + save conversation turn to memory."""
    result = state.get("result")
    query = state.get("query") or ""
    intent = state.get("intent") or ""
    is_cache_hit = state.get("metadata", {}).get("cache_hit", False)

    # Cache: only successful non-chat, non-cached results
    if result and query and intent and not state.get("error") and not is_cache_hit:
        await cache_service.set(intent, query, result)

    # Memory: only store turns for conversational intents with a session
    if result and query and intent in CHAT_INTENTS and state.get("session_id") and not state.get("error"):
        await memory_service.store(
            session_id=state.get("session_id"),
            user_id=state.get("user_id"),
            query=query,
            result=result,
            intent=intent,
        )
    return state


def should_continue(state: WorkflowState) -> str:
    """Determine if workflow should continue."""
    if state.get("error") and "authentication" in state["error"].lower():
        return "end"
    return "continue"


workflow = StateGraph(WorkflowState)

workflow.add_node("authenticate", authenticate_node)
workflow.add_node("load_memory", load_memory_node)
workflow.add_node("retrieve_context", retrieve_context_node)
workflow.add_node("intent_detection", intent_detection_node)
workflow.add_node("planner", planner_node)
workflow.add_node("execute_brain", execute_brain_node)
workflow.add_node("validate", validate_node)
workflow.add_node("store_memory", store_memory_node)

workflow.set_entry_point("authenticate")
workflow.add_edge("authenticate", "load_memory")
workflow.add_edge("load_memory", "retrieve_context")
workflow.add_edge("retrieve_context", "intent_detection")
workflow.add_edge("intent_detection", "planner")
workflow.add_edge("planner", "execute_brain")
workflow.add_edge("execute_brain", "validate")
workflow.add_edge("validate", "store_memory")
workflow.add_edge("store_memory", END)

graph = workflow.compile()
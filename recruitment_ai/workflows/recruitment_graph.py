"""LangGraph recruitment workflow.

START
  → authenticate       (guard: user_id required)
  → load_context        (inject user/resume/job/company profile)
  → load_memory        (inject Redis/DB conversation history)
  → retrieve_context   (RAG: inject Qdrant docs for relevant intents)
  → intent_detection   (classify query → intent)
  → planner            (check Redis cache, stamp planned_brain)
  → execute_brain      (MasterBrain → BrainRouter → selected Brain)
  → validate           (ensure result is always set)
  → store_memory       (persist to Redis cache + conversation memory)
END
"""
from langgraph.graph import StateGraph, END
from recruitment_ai.brains.base import BrainState
from recruitment_ai.workflows.nodes import (
    authenticate_node,
    load_context_node,
    load_memory_node,
    retrieve_context_node,
    planner_node,
    execute_brain_node,
    validate_node,
    store_memory_node,
)
from recruitment_ai.brains.master.intent_classifier import intent_classifier


async def intent_detection_node(state: BrainState) -> BrainState:
    """Detect intent — runs after context/memory/RAG are loaded so classifier has full context."""
    state = await intent_classifier.classify(state)
    state.metadata["classifier"] = state.metadata.get("classifier", "rule")
    return state


def _should_continue(state: BrainState) -> str:
    """Short-circuit on auth failure — skip all remaining nodes."""
    if state.error and "authentication" in state.error.lower():
        return "end"
    return "continue"


def build_graph() -> StateGraph:
    from recruitment_ai.workflows.adapter import to_dict
    workflow = StateGraph(BrainState)

    workflow.add_node("authenticate", to_dict(authenticate_node))
    workflow.add_node("load_context", to_dict(load_context_node))
    workflow.add_node("load_memory", to_dict(load_memory_node))
    workflow.add_node("retrieve_context", to_dict(retrieve_context_node))
    workflow.add_node("intent_detection", to_dict(intent_detection_node))
    workflow.add_node("planner", to_dict(planner_node))
    workflow.add_node("execute_brain", to_dict(execute_brain_node))
    workflow.add_node("validate", to_dict(validate_node))
    workflow.add_node("store_memory", to_dict(store_memory_node))

    workflow.set_entry_point("authenticate")

    workflow.add_conditional_edges(
        "authenticate",
        _should_continue,
        {"continue": "load_context", "end": END},
    )

    workflow.add_edge("load_context", "load_memory")
    workflow.add_edge("load_memory", "retrieve_context")
    workflow.add_edge("retrieve_context", "intent_detection")
    workflow.add_edge("intent_detection", "planner")
    workflow.add_edge("planner", "execute_brain")
    workflow.add_edge("execute_brain", "validate")
    workflow.add_edge("validate", "store_memory")
    workflow.add_edge("store_memory", END)

    return workflow


graph = build_graph().compile()

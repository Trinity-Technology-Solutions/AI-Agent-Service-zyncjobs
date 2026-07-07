"""LangGraph workflow for the recruitment AI platform."""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from recruitment_ai.shared.brain import BrainState
from recruitment_ai.shared.master_brain import master_brain
from recruitment_ai.shared.intent_classifier import intent_classifier


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
    """Load conversation memory."""
    return state


async def retrieve_context_node(state: WorkflowState) -> WorkflowState:
    """Retrieve relevant context from vector store."""
    return state


async def intent_detection_node(state: WorkflowState) -> WorkflowState:
    """Detect intent and route to appropriate brain."""
    brain_state = BrainState(**state)
    brain_state = await intent_classifier.classify(brain_state)
    state["intent"] = brain_state.intent
    return state


async def planner_node(state: WorkflowState) -> WorkflowState:
    """Plan which brain to execute."""
    intent = state.get("intent", "CHAT")
    state["metadata"]["planned_brain"] = intent
    return state


async def execute_brain_node(state: WorkflowState) -> WorkflowState:
    """Execute the selected brain."""
    brain_state = BrainState(**state)
    brain_state = await master_brain.execute(brain_state)
    state.update(brain_state.model_dump())
    return state


async def validate_node(state: WorkflowState) -> WorkflowState:
    """Validate the result."""
    if state.get("error"):
        state["result"] = {"error": state["error"]}
    return state


async def store_memory_node(state: WorkflowState) -> WorkflowState:
    """Store conversation memory."""
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
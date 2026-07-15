"""Tests for LangGraph workflow definition."""
import pytest
from recruitment_ai.workflows.recruitment_graph import graph, build_graph
from recruitment_ai.brains.shared.brain_state import BrainState


def test_graph_is_defined():
    assert graph is not None
    assert build_graph is not None


def test_graph_has_expected_nodes():
    expected = {"authenticate", "load_context", "load_memory", "retrieve_context",
                "intent_detection", "planner", "execute_brain",
                "validate", "store_memory"}
    registered = set(graph.get_graph().nodes.keys())
    for node in expected:
        assert node in registered, f"Node '{node}' not in graph"


def test_initial_state_can_be_created():
    state = BrainState(
        query="Hello",
        user_id="user1",
        user_role="candidate",
    )
    assert state.query == "Hello"
    assert state.user_id == "user1"


@pytest.mark.asyncio
async def test_authenticate_node():
    from recruitment_ai.workflows.nodes.authenticate import authenticate_node
    state = BrainState(user_id="test")
    result = await authenticate_node(state)
    assert result.error is None

    state_no_user = BrainState(user_id=None)
    result = await authenticate_node(state_no_user)
    assert result.error == "Authentication required"


@pytest.mark.asyncio
async def test_intent_detection_node():
    from recruitment_ai.workflows.recruitment_graph import intent_detection_node
    state = BrainState(query="Parse this job description", user_id="test")
    result = await intent_detection_node(state)
    assert result.intent == "JOB_PARSER"


@pytest.mark.asyncio
async def test_validate_node_with_error():
    from recruitment_ai.workflows.nodes.validate import validate_node
    state = BrainState(error="Something went wrong")
    result = await validate_node(state)
    assert result.error == "Something went wrong"

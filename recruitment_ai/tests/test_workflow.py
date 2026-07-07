"""Tests for LangGraph workflow definition."""
import pytest
from recruitment_ai.workflows.recruitment_workflow import workflow, graph
from recruitment_ai.shared.brain import BrainState


def test_workflow_is_defined():
    assert workflow is not None
    assert graph is not None


def test_graph_has_expected_nodes():
    """Check expected nodes exist by looking at registered nodes."""
    expected = {"authenticate", "load_memory", "retrieve_context",
                "intent_detection", "planner", "execute_brain",
                "validate", "store_memory"}
    registered = set(workflow.nodes.keys()) if hasattr(workflow, 'nodes') else set()
    for node in expected:
        assert node in workflow.nodes, f"Node '{node}' not in workflow"


def test_initial_state_can_be_created():
    state = {
        "query": "Hello",
        "intent": None,
        "user_id": "user1",
        "session_id": None,
        "user_role": "candidate",
        "file_content": None,
        "file_type": None,
        "context": None,
        "result": None,
        "error": None,
        "metadata": {},
    }
    assert state["query"] == "Hello"
    assert state["user_id"] == "user1"
    assert state["metadata"] == {}


@pytest.mark.asyncio
async def test_authenticate_node():
    from recruitment_ai.workflows.recruitment_workflow import authenticate_node
    state = {"user_id": "test", "metadata": {}}
    result = await authenticate_node(state)
    assert result.get("error") is None

    state_no_user = {"user_id": None, "metadata": {}}
    result = await authenticate_node(state_no_user)
    assert result.get("error") == "Authentication required"


@pytest.mark.asyncio
async def test_intent_detection_node():
    from recruitment_ai.workflows.recruitment_workflow import intent_detection_node
    state = {
        "query": "Parse this job description",
        "intent": None,
        "user_id": "test",
        "session_id": None,
        "user_role": "candidate",
        "file_content": None,
        "file_type": None,
        "context": None,
        "result": None,
        "error": None,
        "metadata": {},
    }
    result = await intent_detection_node(state)
    assert result["intent"] == "JOB_PARSER"


@pytest.mark.asyncio
async def test_validate_node_with_error():
    from recruitment_ai.workflows.recruitment_workflow import validate_node
    state = {"error": "Something went wrong", "result": None, "metadata": {}}
    result = await validate_node(state)
    assert result["result"] == {"error": "Something went wrong"}


@pytest.mark.asyncio
async def test_should_continue():
    from recruitment_ai.workflows.recruitment_workflow import should_continue
    state_with_auth_error = {"error": "Authentication failed", "metadata": {}}
    state_with_other_error = {"error": "Other error", "metadata": {}}
    state_no_error = {"error": None, "metadata": {}}

    assert should_continue(state_with_auth_error) == "end"
    assert should_continue(state_with_other_error) == "continue"
    assert should_continue(state_no_error) == "continue"

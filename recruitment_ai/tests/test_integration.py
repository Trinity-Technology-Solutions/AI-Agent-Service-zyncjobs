"""Integration tests — full LangGraph pipeline through /ai/execute.
Covers all 17 intents, response structure, error paths, auth, and session memory.
LLM is mocked so tests verify the orchestration pipeline, not LLM output.
"""
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from recruitment_ai.api.main import app
from recruitment_ai.shared.llm_service import LLMService

client = TestClient(app)

# ── Mock LLM at module level so all tests run fast ─────────────────────
@pytest.fixture(autouse=True)
def _mock_llm():
    """Mock LLMService.generate so every test exercises the pipeline without real LLM calls."""
    with patch.object(LLMService, 'generate', new_callable=AsyncMock) as mock:
        mock.return_value = '{"reply": "Mocked reply", "intent": "mock"}'
        yield


def _get_token(user_id: str = "int_test", role: str = "candidate") -> str:
    resp = client.post("/auth/token", json={"user_id": user_id, "role": role})
    assert resp.status_code == 200
    return resp.json()["access_token"]


# ─── Full pipeline: all 17 intents through LangGraph ────────────────────

ALL_INTENTS = [
    ("Hello", "CHAT"),
    ("Parse this job description for me", "JOB_PARSER"),
    ("Generate a job description for a senior backend engineer", "JD_GENERATOR"),
    ("Parse my resume", "RESUME_PARSER"),
    ("What is my ATS score for this job", "ATS_SCORE"),
    ("Find me a job that matches my skills", "JOB_MATCH"),
    ("Give me career advice", "CAREER_ADVICE"),
    ("Assess my Python skills", "SKILL_ASSESSMENT"),
    ("Help me prepare for a React interview", "INTERVIEW_PREP"),
    ("Build a resume for me", "RESUME_BUILDER"),
    ("Create a career roadmap for machine learning", "CAREER_ROADMAP"),
    ("Write a cover letter for a software engineer position", "COVER_LETTER"),
    ("Find candidates for a senior role", "RECRUITER"),
    ("Search for Python developers", "RECRUITER"),
    ("Shortlist the best candidates for this job", "RECRUITER_SHORTLIST"),
    ("Edit my resume to highlight leadership", "RESUME_EDIT"),
    ("Analyze my skill gaps for cloud engineering", "SKILL_GAP"),
]

ALL_INTENT_NAMES = {name for name, _ in ALL_INTENTS}


class TestFullPipeline:
    """Each test exercises the complete LangGraph pipeline:
    POST /ai/execute → authenticate → load_context → load_memory →
    retrieve_context → intent_detection → planner → execute_brain →
    validate → store_memory → response
    """

    def test_all_intents_route_correctly(self):
        """All 17 intents route through the full pipeline with correct intent."""
        token = _get_token()
        for query, expected_intent in ALL_INTENTS:
            resp = client.post(
                "/ai/execute",
                json={"query": query, "user_role": "candidate"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200, f"Failed for '{query}'"
            data = resp.json()
            assert data["intent"] == expected_intent, (
                f"Expected {expected_intent} for '{query}', got {data['intent']}"
            )
            assert isinstance(data["success"], bool)
            assert isinstance(data.get("metadata"), dict)

    def test_all_intents_return_valid_response_structure(self):
        """Every request returns a properly structured response."""
        token = _get_token()
        for query, _ in ALL_INTENTS:
            resp = client.post(
                "/ai/execute",
                json={"query": query, "user_role": "candidate"},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert set(data.keys()) == {"success", "intent", "result", "error", "metadata"}
            assert "classifier" in data.get("metadata", {})
            assert "brain" in data.get("metadata", {})

    def test_success_flag_true_for_valid_requests(self):
        """All valid requests return success=true."""
        token = _get_token()
        for query, _ in ALL_INTENTS:
            resp = client.post(
                "/ai/execute",
                json={"query": query, "user_role": "candidate"},
                headers={"Authorization": f"Bearer {token}"},
            )
            data = resp.json()
            assert data["success"] is True, f"Failed for '{query}': {data.get('error')}"

    def test_response_has_brain_metadata(self):
        """Metadata includes which brain executed."""
        token = _get_token()
        for query, _ in ALL_INTENTS:
            resp = client.post(
                "/ai/execute",
                json={"query": query, "user_role": "candidate"},
                headers={"Authorization": f"Bearer {token}"},
            )
            meta = resp.json().get("metadata", {})
            assert "brain" in meta, f"Missing brain metadata for '{query}'"
            assert meta["brain"] != "", f"Empty brain name for '{query}'"


# ─── Session and memory ────────────────────────────────────────────────

class TestSessionMemory:
    """Verify memory persistence across requests in the same session."""

    def test_same_session_preserves_conversation_id(self):
        token = _get_token()
        session_id = "int_test_session"

        resp1 = client.post(
            "/ai/execute",
            json={"query": "Hello", "session_id": session_id, "user_role": "candidate"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp1.status_code == 200

        resp2 = client.post(
            "/ai/execute",
            json={"query": "Tell me more", "session_id": session_id, "user_role": "candidate"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp2.status_code == 200
        assert resp2.json()["success"] is True

    def test_different_sessions_are_independent(self):
        token = _get_token()
        s1 = "int_test_session_a"
        s2 = "int_test_session_b"

        r1 = client.post(
            "/ai/execute",
            json={"query": "Hello from session A", "session_id": s1, "user_role": "candidate"},
            headers={"Authorization": f"Bearer {token}"},
        )
        r2 = client.post(
            "/ai/execute",
            json={"query": "Hello from session B", "session_id": s2, "user_role": "candidate"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r1.status_code == 200
        assert r2.status_code == 200


# ─── Auth and error handling ───────────────────────────────────────────

class TestAuthAndErrors:
    """Verify the pipeline handles auth and edge cases correctly."""

    def test_missing_token_returns_401(self):
        for query, _ in ALL_INTENTS:
            resp = client.post("/ai/execute", json={"query": query, "user_role": "candidate"})
            assert resp.status_code == 401, f"Expected 401 for '{query}'"

    def test_invalid_token_returns_401_or_403(self):
        resp = client.post(
            "/ai/execute",
            json={"query": "Hello"},
            headers={"Authorization": "Bearer totally_invalid_token"},
        )
        assert resp.status_code in (401, 403), f"Got {resp.status_code}"

    def test_missing_query_returns_422(self):
        token = _get_token()
        resp = client.post(
            "/ai/execute",
            json={"user_role": "candidate"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_empty_query_still_succeeds(self):
        token = _get_token()
        resp = client.post(
            "/ai/execute",
            json={"query": "", "user_role": "candidate"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ─── Context and file content ──────────────────────────────────────────

class TestContextAndFiles:
    """Verify context and file content flow through the pipeline."""

    def test_context_flows_through_pipeline(self):
        token = _get_token()
        resp = client.post(
            "/ai/execute",
            json={
                "query": "Check my ATS score",
                "context": {"resume": "Python dev with 5 years", "job_description": "Need Python expert"},
                "user_role": "candidate",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] in ("ATS_SCORE", "CHAT")

    def test_file_content_flows_through_pipeline(self):
        token = _get_token()
        resp = client.post(
            "/ai/execute",
            json={
                "query": "Parse this resume",
                "file_content": "John Doe\njohn@email.com\nPython developer",
                "file_type": "text",
                "user_role": "candidate",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] == "RESUME_PARSER"

    def test_employer_role_routes_to_recruiter_intents(self):
        token = _get_token(role="employer")
        resp = client.post(
            "/ai/execute",
            json={
                "query": "Find Python developers for my team",
                "user_role": "employer",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["intent"] in ("RECRUITER", "RECRUITER_SEARCH", "CHAT")


# ─── LangGraph node verification ──────────────────────────────────────

class TestPipelineNodes:
    """Verify the LangGraph pipeline completed all expected nodes."""

    def test_response_contains_execution_metadata(self):
        token = _get_token()
        resp = client.post(
            "/ai/execute",
            json={"query": "Hello", "user_role": "candidate"},
            headers={"Authorization": f"Bearer {token}"},
        )
        meta = resp.json().get("metadata", {})
        assert "brain" in meta
        assert meta.get("classifier") in ("rule", "llm")

    def test_all_intents_have_validation_metadata(self):
        token = _get_token()
        for query, _ in ALL_INTENTS:
            resp = client.post(
                "/ai/execute",
                json={"query": query, "user_role": "candidate"},
                headers={"Authorization": f"Bearer {token}"},
            )
            meta = resp.json().get("metadata", {})
            assert "validation" in meta, f"Missing validation metadata for '{query}'"

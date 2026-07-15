"""Integration tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from recruitment_ai.api.main import app
from recruitment_ai.shared.llm_service import LLMService
from recruitment_ai.vector.store import vector_store
from unittest.mock import AsyncMock, patch

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama" in data


def test_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "brains" in data
    assert "knowledge_chunks" in data
    assert len(data["brains"]) >= 12


def test_knowledge_endpoint():
    response = client.get("/knowledge/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_chunks" in data
    assert "backend" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.content.decode()
    # prometheus text or JSON fallback
    assert ("chunks" in content) or ("python_gc" in content) or ("# HELP" in content)


def test_create_token():
    response = client.post("/auth/token", json={"user_id": "test_user", "role": "candidate"})
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_create_token_with_email():
    response = client.post("/auth/token", json={
        "user_id": "emp1", "role": "employer", "email": "emp@example.com"
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data["access_token"]) > 20


def test_execute_without_token_returns_401():
    response = client.post("/ai/execute", json={"query": "Hello"})
    assert response.status_code == 401
    data = response.json()
    assert "detail" in data


def test_execute_with_token():
    # Get token first
    token_resp = client.post("/auth/token", json={"user_id": "test_user", "role": "candidate"})
    token = token_resp.json()["access_token"]

    response = client.post(
        "/ai/execute",
        json={"query": "Hello, ZyncJobs!", "user_role": "candidate"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "intent" in data


def test_execute_with_session():
    token_resp = client.post("/auth/token", json={"user_id": "test_user", "role": "candidate"})
    token = token_resp.json()["access_token"]

    response = client.post(
        "/ai/execute",
        json={"query": "Parse this job", "session_id": "sess_123", "user_role": "candidate"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["intent"] == "JOB_PARSER"


def test_execute_with_context():
    token_resp = client.post("/auth/token", json={"user_id": "test_user", "role": "candidate"})
    token = token_resp.json()["access_token"]

    response = client.post(
        "/ai/execute",
        json={
            "query": "Check ATS",
            "context": {"resume": "Python dev 3yrs", "job_description": "Need Python"},
            "user_role": "candidate",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    # Should route to ATS_SCORE
    assert response.json()["intent"] == "ATS_SCORE"


def test_execute_with_file_content():
    token_resp = client.post("/auth/token", json={"user_id": "test_user", "role": "candidate"})
    token = token_resp.json()["access_token"]

    response = client.post(
        "/ai/execute",
        json={
            "query": "Parse this resume",
            "file_content": "John Doe\njohn@email.com\nPython developer",
            "file_type": "text",
            "user_role": "candidate",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["intent"] == "RESUME_PARSER"


def test_invalid_token():
    response = client.post(
        "/ai/execute",
        json={"query": "Hello"},
        headers={"Authorization": "Bearer invalid_token_here"},
    )
    # Should return 401 or 403
    assert response.status_code in [401, 403]


def test_malformed_execute_request():
    """Missing required 'query' field should return 422."""
    token_resp = client.post("/auth/token", json={"user_id": "test_user", "role": "candidate"})
    token = token_resp.json()["access_token"]

    response = client.post(
        "/ai/execute",
        json={"user_role": "candidate"},  # missing query
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


def test_execute_all_intents():
    """Quick smoke test that each intent routes correctly."""
    token_resp = client.post("/auth/token", json={"user_id": "test_user", "role": "candidate"})
    token = token_resp.json()["access_token"]

    tests = [
        ("Parse this job description", "JOB_PARSER"),
        ("Generate a job description", "JD_GENERATOR"),
        ("Parse my resume", "RESUME_PARSER"),
        ("What is my ATS score", "ATS_SCORE"),
        ("Find me a job", "JOB_MATCH"),
        ("Career advice", "CAREER_ADVICE"),
        ("Skill assessment for Python", "SKILL_ASSESSMENT"),
        ("Interview preparation", "CAREER_ADVICE"),
        ("Build a resume", "RESUME_BUILDER"),
        ("Find candidates", "RECRUITER"),
        ("Shortlist candidates", "RECRUITER_SHORTLIST"),
    ]

    for query, expected_intent in tests:
        response = client.post(
            "/ai/execute",
            json={"query": query, "user_role": "candidate"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, f"Failed for '{query}'"
        data = response.json()
        assert data["intent"] == expected_intent, f"Expected {expected_intent} for '{query}', got {data['intent']}"

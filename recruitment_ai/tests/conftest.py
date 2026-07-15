"""Shared test fixtures and mocks."""
import os
os.environ["RATE_LIMIT_ENABLED"] = "false"

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from recruitment_ai.brains.base import BrainState
from recruitment_ai.shared.llm_service import LLMService


@pytest.fixture
def brain_state():
    """Basic BrainState fixture."""
    return BrainState(
        query="test query",
        user_id="test_user",
        session_id="test_session",
        user_role="candidate",
    )


@pytest.fixture
def employer_state():
    """BrainState for employer/recruiter tests."""
    return BrainState(
        query="Find Python developers",
        user_id="emp_user",
        session_id="emp_session",
        user_role="employer",
    )


@pytest.fixture
def mock_ollama():
    """Mock LLMService.generate to return a predictable response."""
    patcher = patch.object(LLMService, 'generate', new_callable=AsyncMock)
    mock = patcher.start()
    mock.return_value = '{"result": "mocked response"}'
    yield mock
    patcher.stop()


@pytest.fixture
def mock_ollama_failure():
    """Mock LLMService.generate to raise an exception (tests fallback)."""
    patcher = patch.object(LLMService, 'generate', new_callable=AsyncMock)
    mock = patcher.start()
    mock.side_effect = Exception("Ollama unavailable")
    yield mock
    patcher.stop()


@pytest.fixture
def mock_ollama_jd():
    """Mock LLMService.generate returning a job description."""
    patcher = patch.object(LLMService, 'generate', new_callable=AsyncMock)
    mock = patcher.start()
    mock.return_value = """# Software Engineer at ZyncJobs

## About the Company
ZyncJobs is a leading technology company.

## About the Role
We are looking for a mid Software Engineer.

## Key Responsibilities
Develop features, Write tests

## Required Qualifications
3+ years experience

## How to Apply
Submit your resume."""
    yield mock
    patcher.stop()


@pytest.fixture
def mock_ollama_ats():
    """Mock LLMService.generate returning ATS analysis JSON."""
    patcher = patch.object(LLMService, 'generate', new_callable=AsyncMock)
    mock = patcher.start()
    mock.return_value = '{"ats_score": 85, "keyword_match": {"matched": ["python", "react"], "missing": ["docker"], "match_percentage": 67}, "formatting_score": 80, "section_completeness": 90, "experience_relevance": 75, "suggestions": ["Add missing skill: docker"], "passes_ats": true}'
    yield mock
    patcher.stop()


@pytest.fixture(autouse=True)
def mock_backend_client():
    """Mock BackendClient globally so tests never make real HTTP calls."""
    patcher = patch('recruitment_ai.services.backend_client.backend_client')
    mock = patcher.start()

    async def return_none(*args, **kwargs):
        return None

    async def return_empty_list(*args, **kwargs):
        return []

    mock.get_user.side_effect = return_none
    mock.get_resume.side_effect = return_none
    mock.get_company.side_effect = return_none
    mock.get_job.side_effect = return_none
    mock.get_assessment.side_effect = return_none
    mock.get_conversation_history.side_effect = return_empty_list
    mock.search_jobs.side_effect = return_empty_list
    mock.search_candidates.side_effect = return_empty_list
    mock.health_check.side_effect = return_none

    yield mock
    patcher.stop()


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for ChatbotBrain tests."""
    patcher = patch('recruitment_ai.vector.store.vector_store')
    mock = patcher.start()

    class MockDoc:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    mock.search = AsyncMock(return_value=[
        MockDoc(
            text="ZyncJobs is an AI recruitment platform.",
            metadata={"title": "About ZyncJobs", "url": "https://zyncjobs.ai/about", "category": "platform"},
        )
    ])
    mock.count = 579
    yield mock
    patcher.stop()


@pytest.fixture
def mock_vector_store_empty():
    """Mock VectorStore returning no results."""
    patcher = patch('recruitment_ai.vector.store.vector_store')
    mock = patcher.start()
    mock.search = AsyncMock(return_value=[])
    mock.count = 0
    yield mock
    patcher.stop()

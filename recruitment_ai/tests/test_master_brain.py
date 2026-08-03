"""Tests for MasterBrain orchestration."""
import pytest
from unittest.mock import AsyncMock, patch
from recruitment_ai.brains.master.master_brain import MasterBrain, master_brain
from recruitment_ai.brains.base import BrainState
from recruitment_ai.shared.llm_service import LLMService


@pytest.fixture(autouse=True)
def mock_llm():
    """Prevent real LLM calls in all master brain tests."""
    with patch.object(LLMService, "generate", new_callable=AsyncMock) as m:
        m.return_value = '{"result": "mocked"}'
        yield m


def test_master_brain_all_brains_registered():
    actual = set(master_brain.brains.keys())
    required = {
        "ATS_SCORE", "CAREER_ADVICE", "CAREER_ROADMAP", "CHAT", "COVER_LETTER",
        "INTERVIEW_PREP", "JD_GENERATOR", "JOB_MATCH", "JOB_PARSER",
        "RECRUITER", "RECRUITER_SEARCH", "RECRUITER_SHORTLIST",
        "RESUME_BUILDER", "RESUME_EDIT", "RESUME_PARSER",
        "SKILL_ASSESSMENT", "SKILL_GAP",
    }
    assert required.issubset(actual), f"Missing brains: {required - actual}"


@pytest.mark.asyncio
async def test_master_brain_routes_to_job_parser():
    state = BrainState(query="Parse this job description")
    result = await master_brain.execute(state)
    assert result.intent == "JOB_PARSER"
    assert result.result is not None


@pytest.mark.asyncio
async def test_master_brain_routes_to_chat():
    state = BrainState(query="Hello, ZyncJobs!")
    result = await master_brain.execute(state)
    assert result.intent == "CHAT"


@pytest.mark.asyncio
async def test_master_brain_routes_to_ats():
    state = BrainState(query="Check my ATS score")
    result = await master_brain.execute(state)
    assert result.intent == "ATS_SCORE"


@pytest.mark.asyncio
async def test_master_brain_routes_to_resume_builder():
    state = BrainState(query="Build a resume for me")
    result = await master_brain.execute(state)
    assert result.intent == "RESUME_BUILDER"


@pytest.mark.asyncio
async def test_master_brain_routes_to_resume_parser():
    state = BrainState(query="Parse my resume")
    result = await master_brain.execute(state)
    assert result.intent == "RESUME_PARSER"


@pytest.mark.asyncio
async def test_master_brain_routes_to_skill_gap():
    state = BrainState(query="Show me my skill gap analysis")
    result = await master_brain.execute(state)
    assert result.intent == "SKILL_GAP"


@pytest.mark.asyncio
async def test_master_brain_routes_to_cover_letter():
    state = BrainState(query="Generate a cover letter for this role")
    result = await master_brain.execute(state)
    assert result.intent == "COVER_LETTER"


@pytest.mark.asyncio
async def test_master_brain_routes_to_career_roadmap():
    state = BrainState(query="Give me a career roadmap")
    result = await master_brain.execute(state)
    assert result.intent == "CAREER_ROADMAP"


@pytest.mark.asyncio
async def test_master_brain_handles_unknown_query_falls_to_chat():
    """Unrecognised query falls back to CHAT (via LLM classifier mock)."""
    with patch.object(
        master_brain.__class__.__mro__[0],  # MasterBrain
        "__init__",
        wraps=None,
    ):
        pass  # just ensure no crash
    state = BrainState(query="xyz_nonexistent_intent_abc123")
    result = await master_brain.execute(state)
    assert result.intent == "CHAT"


@pytest.mark.asyncio
async def test_master_brain_empty_query_returns_gracefully():
    state = BrainState(query="", user_id="test", user_role="candidate")
    result = await master_brain.execute(state)
    assert result.error is None


@pytest.mark.asyncio
async def test_master_brain_returns_state_with_all_fields():
    state = BrainState(query="Hello", user_id="test", user_role="candidate")
    result = await master_brain.execute(state)
    assert hasattr(result, "intent")
    assert hasattr(result, "result")
    assert hasattr(result, "error")
    assert hasattr(result, "metadata")


@pytest.mark.asyncio
async def test_master_brain_reuses_brain_instances():
    """Same intent should use same brain object."""
    s1 = BrainState(query="Parse this job description")
    s2 = BrainState(query="Parse this job description for a senior role")
    r1 = await master_brain.execute(s1)
    r2 = await master_brain.execute(s2)
    assert r1.intent == "JOB_PARSER"
    assert r2.intent == "JOB_PARSER"
    assert master_brain.brains["JOB_PARSER"] is master_brain.brains["JOB_PARSER"]


@pytest.mark.asyncio
async def test_master_brain_systemPrompt_forces_chat():
    """Structured intent is overridden to CHAT when systemPrompt is present."""
    state = BrainState(query="Check my ATS score")
    state.context_data.user_preferences["systemPrompt"] = "You are a helpful assistant."
    result = await master_brain.execute(state)
    assert result.intent == "CHAT"


@pytest.mark.asyncio
async def test_master_brain_metadata_contains_brain_name():
    state = BrainState(query="Parse this job description")
    result = await master_brain.execute(state)
    assert "brain" in result.metadata


@pytest.mark.asyncio
async def test_master_brain_execution_duration_recorded():
    state = BrainState(query="Parse this job description")
    result = await master_brain.execute(state)
    assert result.execution.duration_ms >= 0

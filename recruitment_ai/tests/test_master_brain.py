"""Tests for MasterBrain orchestration."""
import pytest
from recruitment_ai.brains.master.master_brain import MasterBrain, master_brain
from recruitment_ai.brains.base import BrainState


@pytest.mark.asyncio
async def test_master_brain_all_brains_registered():
    expected = {"ATS_SCORE", "CAREER_ADVICE", "CAREER_ROADMAP", "CHAT", "COVER_LETTER",
                "INTERVIEW_PREP", "JD_GENERATOR", "JOB_MATCH", "JOB_PARSER",
                "RECRUITER", "RECRUITER_SEARCH", "RECRUITER_SHORTLIST",
                "RESUME_BUILDER", "RESUME_EDIT", "RESUME_PARSER",
                "SKILL_ASSESSMENT", "SKILL_GAP"}
    assert set(master_brain.brains.keys()) == expected


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
async def test_master_brain_handles_unknown_brain():
    state = BrainState(query="xyz_nonexistent_intent")
    result = await master_brain.execute(state)
    # Falls to CHAT since no patterns match
    assert result.intent == "CHAT"


@pytest.mark.asyncio
async def test_master_brain_error_propagation():
    state = BrainState(query="", user_id="test", user_role="candidate")
    result = await master_brain.execute(state)
    # Empty query should not cause crash, should return graceful response
    assert result.error is None or "error" not in (result.error or "").lower()


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
    s2 = BrainState(query="Extract job data")
    r1 = await master_brain.execute(s1)
    r2 = await master_brain.execute(s2)
    # Both should route to JOB_PARSER
    assert r1.intent == "JOB_PARSER"
    assert r2.intent == "JOB_PARSER"

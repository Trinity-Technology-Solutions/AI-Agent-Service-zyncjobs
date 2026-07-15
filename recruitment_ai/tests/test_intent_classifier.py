"""Tests for IntentClassifier."""
import pytest
from recruitment_ai.brains.master.intent_classifier import IntentClassifier, intent_classifier
from recruitment_ai.brains.master.intent_classifier import INTENT_PATTERNS
from recruitment_ai.brains.base import BrainState


@pytest.mark.asyncio
async def test_classify_job_parser():
    state = BrainState(query="Parse this job description")
    result = await intent_classifier.classify(state)
    assert result.intent == "JOB_PARSER"


@pytest.mark.asyncio
async def test_classify_jd_generator():
    state = BrainState(query="Generate a job description for a senior developer")
    result = await intent_classifier.classify(state)
    assert result.intent == "JD_GENERATOR"


@pytest.mark.asyncio
async def test_classify_resume_parser():
    state = BrainState(query="Parse this resume for me")
    result = await intent_classifier.classify(state)
    assert result.intent == "RESUME_PARSER"


@pytest.mark.asyncio
async def test_classify_ats_score():
    state = BrainState(query="What is my ATS score?")
    result = await intent_classifier.classify(state)
    assert result.intent == "ATS_SCORE"


@pytest.mark.asyncio
async def test_classify_job_match():
    state = BrainState(query="Find me a job match")
    result = await intent_classifier.classify(state)
    assert result.intent == "JOB_MATCH"


@pytest.mark.asyncio
async def test_classify_career_advice():
    state = BrainState(query="Give me career advice")
    result = await intent_classifier.classify(state)
    assert result.intent == "CAREER_ADVICE"


@pytest.mark.asyncio
async def test_classify_skill_assessment():
    state = BrainState(query="Take a skill assessment for Python")
    result = await intent_classifier.classify(state)
    assert result.intent == "SKILL_ASSESSMENT"


@pytest.mark.asyncio
async def test_classify_interview_prep():
    state = BrainState(query="Help with interview preparation")
    result = await intent_classifier.classify(state)
    assert result.intent == "CAREER_ADVICE"


@pytest.mark.asyncio
async def test_classify_resume_builder():
    state = BrainState(query="Build a resume for me")
    result = await intent_classifier.classify(state)
    assert result.intent == "RESUME_BUILDER"


@pytest.mark.asyncio
async def test_classify_recruiter():
    state = BrainState(query="Find candidates for my job")
    result = await intent_classifier.classify(state)
    assert result.intent == "RECRUITER"


@pytest.mark.asyncio
async def test_classify_shortlist():
    state = BrainState(query="Shortlist the top candidates")
    result = await intent_classifier.classify(state)
    assert result.intent == "RECRUITER_SHORTLIST"


@pytest.mark.asyncio
async def test_classify_chat_greeting():
    state = BrainState(query="Hello, how can I use ZyncJobs?")
    result = await intent_classifier.classify(state)
    assert result.intent == "CHAT"


@pytest.mark.asyncio
async def test_classify_empty_query():
    state = BrainState(query="")
    result = await intent_classifier.classify(state)
    assert result.intent == "CHAT"


@pytest.mark.asyncio
async def test_classify_whitespace_query():
    state = BrainState(query="   ")
    result = await intent_classifier.classify(state)
    assert result.intent == "CHAT"


@pytest.mark.asyncio
async def test_classify_unknown_falls_to_chat():
    state = BrainState(query="asdfghjkl")
    result = await intent_classifier.classify(state)
    assert result.intent == "CHAT"


@pytest.mark.asyncio
async def test_classify_sync():
    assert intent_classifier.classify_sync("Parse this job") == "JOB_PARSER"
    assert intent_classifier.classify_sync("") == "CHAT"
    assert intent_classifier.classify_sync("Random text") == "CHAT"


@pytest.mark.asyncio
async def test_classify_all_intents_have_patterns():
    """Every registered intent must have at least one pattern."""
    imports = ["ATS_SCORE", "CAREER_ADVICE", "CAREER_ROADMAP", "CHAT", "COVER_LETTER",
               "INTERVIEW_PREP", "JD_GENERATOR", "JOB_MATCH", "JOB_PARSER",
               "RECRUITER", "RECRUITER_SEARCH", "RECRUITER_SHORTLIST",
               "RESUME_BUILDER", "RESUME_EDIT", "RESUME_PARSER",
               "SKILL_ASSESSMENT", "SKILL_GAP"]
    for intent in imports:
        assert intent in INTENT_PATTERNS, f"{intent} missing from patterns"
        assert len(INTENT_PATTERNS[intent]) > 0, f"{intent} has no patterns"

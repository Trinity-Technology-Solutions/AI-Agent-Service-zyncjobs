"""Tests for all brains — focused on fallback logic when LLM is unavailable."""
import pytest
from recruitment_ai.brains.base import BrainState
from recruitment_ai.brains.chatbot.chatbot_brain import ChatbotBrain
from recruitment_ai.brains.employer.job_parser_brain import JobParserBrain
from recruitment_ai.brains.employer.jd_generator_brain import JDGeneratorBrain
from recruitment_ai.brains.candidate.resume_parser_brain import ResumeParserBrain
from recruitment_ai.brains.candidate.ats_brain import ATSBrain
from recruitment_ai.brains.candidate.job_matching_brain import JobMatchingBrain
from recruitment_ai.brains.candidate.career_brain import CareerBrain
from recruitment_ai.brains.employer.recruiter_brain import RecruiterBrain


chatbot_brain = ChatbotBrain()
job_parser_brain = JobParserBrain()
jd_generator_brain = JDGeneratorBrain()
resume_parser_brain = ResumeParserBrain()
ats_brain = ATSBrain()
job_matching_brain = JobMatchingBrain()
career_brain = CareerBrain()
recruiter_brain = RecruiterBrain()


# ─── ChatbotBrain ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chatbot_empty_query():
    state = BrainState(query="")
    result = await chatbot_brain.run(state)
    assert result.response is not None
    assert "reply" in result.response
    assert "How can I help" in result.response["reply"]


@pytest.mark.asyncio
async def test_chatbot_no_vector_results():
    state = BrainState(query="Something unknown")
    result = await chatbot_brain.run(state)
    assert result.response is not None
    assert "reply" in result.response


@pytest.mark.asyncio
async def test_chatbot_with_context(mock_vector_store):
    state = BrainState(query="What is ZyncJobs?")
    result = await chatbot_brain.run(state)
    assert result.response is not None
    assert "reply" in result.response or "error" in result.response


@pytest.mark.asyncio
async def test_chatbot_context_building():
    docs = [
        {"text": "ZyncJobs is great.", "title": "About"},
        {"text": "Features include AI matching.", "title": "Features"},
    ]
    context = chatbot_brain._build_context(docs)
    assert "ZyncJobs is great." in context
    assert "Features include AI matching." in context
    assert "[About]" in context
    assert "[Features]" in context


@pytest.mark.asyncio
async def test_chatbot_citations_building():
    docs = [
        {"text": "text", "title": "T", "url": "/u"},
    ]
    citations = chatbot_brain._build_citations(docs)
    assert len(citations) == 1
    assert citations[0]["title"] == "T"
    assert citations[0]["url"] == "/u"


# ─── JobParserBrain ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_parser_empty():
    state = BrainState(query="")
    result = await job_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("error") == "No job description provided"


@pytest.mark.asyncio
async def test_job_parser_llm_success(mock_ollama):
    state = BrainState(query="Senior Python Developer at Acme Corp")
    result = await job_parser_brain.run(state)
    assert result.response is not None
    assert result.metadata.get("parser") == "llm"


@pytest.mark.asyncio
async def test_job_parser_fallback(mock_ollama_failure):
    state = BrainState(query="We need a Senior Python Developer with 5+ years of experience. Skills: Python, Django, PostgreSQL")
    result = await job_parser_brain.run(state)
    assert result.response is not None
    assert result.metadata.get("parser") == "fallback"
    assert "title" in result.response
    assert "skills_required" in result.response
    assert "python" in [s.lower() for s in result.response["skills_required"]]


@pytest.mark.asyncio
async def test_job_parser_fallback_title_extracted():
    state = BrainState(query="Senior Python Developer\nWe need someone experienced.")
    result = await job_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("title") == "Senior Python Developer"


# ─── JDGeneratorBrain ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jd_generator_empty():
    state = BrainState(query="", context={})
    result = await jd_generator_brain.run(state)
    assert result.response is not None
    assert result.response.get("error") == "No job details provided"


@pytest.mark.asyncio
async def test_jd_generator_with_context(mock_ollama_jd):
    state = BrainState(query="", context={"title": "Backend Engineer", "company": "ZyncJobs"})
    result = await jd_generator_brain.run(state)
    assert result.response is not None
    assert "job_description" in result.response


@pytest.mark.asyncio
async def test_jd_generator_fallback(mock_ollama_failure):
    state = BrainState(query="Senior Dev", context={"title": "Backend Engineer"})
    result = await jd_generator_brain.run(state)
    assert result.response is not None
    assert result.response.get("fallback") is True
    assert "Backend Engineer" in result.response["job_description"]


@pytest.mark.asyncio
async def test_jd_generator_template_fallback():
    params = jd_generator_brain._template_fallback({
        "title": "DevOps Engineer", "company": "TestCo",
        "location": "Remote", "experience_level": "mid",
        "skills": "Python, Docker",
    })
    assert "DevOps Engineer" in params
    assert "TestCo" in params


# ─── ResumeParserBrain ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resume_parser_empty():
    state = BrainState(query="")
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("error") == "No resume content provided"


@pytest.mark.asyncio
async def test_resume_parser_llm_success(mock_ollama):
    state = BrainState(query="John Doe\njohn@email.com\nPython developer with 5 years experience")
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.metadata.get("parser") == "llm"


@pytest.mark.asyncio
async def test_resume_parser_fallback(mock_ollama_failure):
    state = BrainState(query="John Doe\njohn@email.com\nPython developer skilled in Django, React")
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.metadata.get("parser") == "fallback"
    assert "name" in result.response
    assert "email" in result.response
    assert "skills" in result.response


@pytest.mark.asyncio
async def test_resume_parser_fallback_extract_email():
    state = BrainState(query="Contact me at john@example.com")
    result = await resume_parser_brain.run(state)
    assert result.response.get("email") == "john@example.com"


@pytest.mark.asyncio
async def test_resume_parser_fallback_extract_phone():
    state = BrainState(query="Phone: +1 (555) 123-4567")
    result = await resume_parser_brain.run(state)
    assert result.response.get("phone", "") != ""


@pytest.mark.asyncio
async def test_resume_parser_fallback_extract_name():
    state = BrainState(query="John Doe\nSome experience")
    result = await resume_parser_brain.run(state)
    assert result.response.get("name") == "John Doe"


# ─── ATSBrain ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ats_brain_no_resume():
    state = BrainState(query="")
    result = await ats_brain.run(state)
    assert result.response is not None
    assert result.response.get("error") == "No resume provided"


@pytest.mark.asyncio
async def test_ats_brain_rule_based():
    state = BrainState(
        query="Python developer with React experience",
        context={"job_description": "Looking for Python, React, Docker expert"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert "ats_score" in result.response
    assert result.response.get("keyword_match", {}).get("matched") is not None


@pytest.mark.asyncio
async def test_ats_brain_fallback(mock_ollama_failure):
    state = BrainState(
        query="Python and React developer",
        context={"job_description": "Need Python, Docker, AWS skills"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert result.metadata.get("model") == "rule_based"
    assert "ats_score" in result.response


# ─── JobMatchingBrain ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_matching_missing_fields():
    state = BrainState(query="")
    result = await job_matching_brain.run(state)
    assert result.response is not None
    assert "Both candidate profile and job requirements required" in result.response.get("error", "")


@pytest.mark.asyncio
async def test_job_matching_fallback(mock_ollama_failure):
    state = BrainState(
        query="Python developer with 3 years experience",
        context={"job_requirements": "Need Python, React, Docker. 2+ years experience."},
    )
    result = await job_matching_brain.run(state)
    assert result.response is not None
    assert result.metadata.get("model") == "rule_based"
    assert "match_score" in result.response
    assert "recommendation" in result.response


@pytest.mark.asyncio
async def test_job_matching_rule_based_high_match(mock_ollama_failure):
    state = BrainState(
        query="Python, React, Docker, AWS developer with 5 years",
        context={"job_requirements": "Python, React, Docker, AWS required. 3+ years."},
    )
    result = await job_matching_brain.run(state)
    assert result.response is not None
    assert result.response.get("match_score", 0) >= 60
    assert result.response.get("recommendation") in ["strong_match", "good_match"]


@pytest.mark.asyncio
async def test_job_matching_rule_based_low_match(mock_ollama_failure):
    state = BrainState(
        query="Java developer",
        context={"job_requirements": "Python, React, Docker required. 3+ years."},
    )
    result = await job_matching_brain.run(state)
    assert result.response is not None
    assert result.response.get("match_score", 100) < 60


# ─── CareerBrain ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_career_advice_fallback(mock_ollama_failure):
    state = BrainState(
        query="Career advice",
        intent="CAREER_ADVICE",
    )
    result = await career_brain.run(state)
    assert result.response is not None
    assert "reply" in result.response or "intent" in result.response


@pytest.mark.asyncio
async def test_skill_assessment_fallback(mock_ollama_failure):
    state = BrainState(
        query="Test my Python skills",
        intent="SKILL_ASSESSMENT",
        context={"skill": "Python", "level": "intermediate"},
    )
    result = await career_brain.run(state)
    assert result.response is not None
    assert "questions" in result.response
    assert result.response["questions"] == []


@pytest.mark.asyncio
async def test_interview_prep_fallback(mock_ollama_failure):
    state = BrainState(
        query="Interview prep for React dev",
        intent="INTERVIEW_PREP",
        context={"role": "React Developer", "level": "mid"},
    )
    result = await career_brain.run(state)
    assert result.response is not None
    assert "questions" in result.response
    assert result.response["topics_to_review"] == []


@pytest.mark.asyncio
async def test_resume_builder_fallback(mock_ollama_failure):
    state = BrainState(
        query="Build my resume",
        intent="RESUME_BUILDER",
        context={
            "personal_info": {"name": "John"},
            "experience": [{"company": "Acme"}],
            "skills": {"technical": ["Python"]},
            "target_role": "Senior Dev",
        },
    )
    result = await career_brain.run(state)
    assert result.response is not None
    assert "summary" in result.response


@pytest.mark.asyncio
async def test_career_unknown_intent_defaults_to_advice(mock_ollama_failure):
    state = BrainState(query="Something", intent="UNKNOWN")
    result = await career_brain.run(state)
    assert result.response is not None


# ─── RecruiterBrain ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recruiter_search_fallback(mock_ollama_failure):
    state = BrainState(
        query="Find Python developers",
        user_id="emp_user",
        user_role="employer",
        context={"skills": ["Python", "Django"]},
        context_data={"job": {"title": "Software Engineer", "description": "Python developer needed"}},
    )
    result = await recruiter_brain.run(state)
    assert result.response is not None
    assert result.metadata.get("fallback") is True
    assert "search_strategy" in result.response
    assert "Python" in result.response["search_strategy"]


@pytest.mark.asyncio
async def test_recruiter_shortlist_fallback(mock_ollama_failure):
    state = BrainState(
        query="Shortlist the best candidates",
        context={
            "job_requirements": "Python developer",
            "candidates": [{"name": "Alice"}, {"name": "Bob"}],
        },
        context_data={"job": {"title": "Software Engineer"}},
    )
    result = await recruiter_brain.run(state)
    assert result.response is not None
    assert "shortlisted" in result.response


@pytest.mark.asyncio
async def test_recruiter_routes_to_search_by_default(mock_ollama_failure):
    state = BrainState(
        query="Find candidates",
        context_data={"job": {"title": "Engineer", "description": "Python dev needed"}},
    )
    result = await recruiter_brain.run(state)
    assert result.response is not None
    assert "search_strategy" in result.response or "recommended_filters" in result.response


@pytest.mark.asyncio
async def test_recruiter_shortlist_route(mock_ollama_failure):
    state = BrainState(
        query="shortlist candidates",
        context={"job_requirements": "Python dev", "candidates": [{"name": "Alice"}]},
        context_data={"job": {"title": "Engineer", "description": "Python"}},
    )
    result = await recruiter_brain.run(state)
    assert result.response is not None

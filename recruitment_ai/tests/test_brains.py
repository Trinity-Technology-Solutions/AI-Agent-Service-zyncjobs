"""Tests for all brains — focused on fallback logic when Ollama is unavailable."""
import pytest
from recruitment_ai.shared.brain import BrainState
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
    assert result.result is not None
    assert "reply" in result.result
    assert "How can I help" in result.result["reply"]


@pytest.mark.asyncio
async def test_chatbot_no_vector_results():
    from unittest.mock import AsyncMock, patch
    with patch('recruitment_ai.brains.chatbot.chatbot_brain.vector_store') as mock_vs:
        mock_vs.search = AsyncMock(return_value=[])
        state = BrainState(query="Something unknown")
        result = await chatbot_brain.run(state)
    assert result.result is not None
    assert "couldn't find" in result.result["reply"].lower()


@pytest.mark.asyncio
async def test_chatbot_with_context(mock_vector_store):
    state = BrainState(query="What is ZyncJobs?")
    result = await chatbot_brain.run(state)
    assert result.result is not None
    assert "reply" in result.result or "error" in result.result


@pytest.mark.asyncio
async def test_chatbot_context_building():
    from recruitment_ai.vector.store import vector_store
    from unittest.mock import MagicMock

    class MockDoc:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    docs = [
        MockDoc("ZyncJobs is great.", {"title": "About", "url": "/about", "category": "platform"}),
        MockDoc("Features include AI matching.", {"title": "Features", "url": "/features", "category": "tech"}),
    ]
    context = chatbot_brain._build_context(docs)
    assert "ZyncJobs is great." in context
    assert "Features include AI matching." in context
    assert "[About]" in context
    assert "[Features]" in context


@pytest.mark.asyncio
async def test_chatbot_sources_extraction():
    from recruitment_ai.vector.store import vector_store
    from unittest.mock import MagicMock

    class MockDoc:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    docs = [MockDoc("text", {"title": "T", "url": "/u", "category": "c"})]
    sources = chatbot_brain._extract_sources(docs)
    assert len(sources) == 1
    assert sources[0]["title"] == "T"
    assert sources[0]["url"] == "/u"


# ─── JobParserBrain ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_parser_empty():
    state = BrainState(query="")
    result = await job_parser_brain.run(state)
    assert result.error == "No job description provided"


@pytest.mark.asyncio
async def test_job_parser_llm_success(mock_ollama):
    state = BrainState(query="Senior Python Developer at Acme Corp")
    result = await job_parser_brain.run(state)
    assert result.result is not None
    assert result.metadata.get("parser") == "llm"


@pytest.mark.asyncio
async def test_job_parser_fallback(mock_ollama_failure):
    state = BrainState(query="We need a Senior Python Developer with 5+ years of experience. Skills: Python, Django, PostgreSQL")
    result = await job_parser_brain.run(state)
    assert result.result is not None
    assert result.metadata.get("parser") == "fallback"
    assert "title" in result.result
    assert "skills_required" in result.result
    assert "python" in [s.lower() for s in result.result["skills_required"]]


@pytest.mark.asyncio
async def test_job_parser_fallback_title_extracted():
    state = BrainState(query="Senior Python Developer\nWe need someone experienced.")
    result = await job_parser_brain.run(state)
    assert result.result is not None
    assert result.result["title"] == "Senior Python Developer"


@pytest.mark.asyncio
async def test_job_parser_fallback_empty_result():
    result = job_parser_brain._empty_result()
    assert result["title"] == ""
    assert result["skills_required"] == []


# ─── JDGeneratorBrain ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jd_generator_empty():
    state = BrainState(query="")
    result = await jd_generator_brain.run(state)
    assert result.error == "No job details provided"


@pytest.mark.asyncio
async def test_jd_generator_with_context(mock_ollama_jd):
    state = BrainState(query="", context={"title": "Backend Engineer", "company": "ZyncJobs"})
    result = await jd_generator_brain.run(state)
    assert result.result is not None
    assert "job_description" in result.result


@pytest.mark.asyncio
async def test_jd_generator_fallback(mock_ollama_failure):
    state = BrainState(query="Senior Dev", context={"title": "Backend Engineer"})
    result = await jd_generator_brain.run(state)
    assert result.result is not None
    assert result.result.get("fallback") is True
    assert "Backend Engineer" in result.result["job_description"]


@pytest.mark.asyncio
async def test_jd_generator_template_fallback():
    state = BrainState(query="", context={"title": "DevOps Engineer", "company": "TestCo"})
    result = await jd_generator_brain.run(state)
    result = await jd_generator_brain.run(state) if not result.result else result
    params = jd_generator_brain._extract_params({"title": "DevOps Engineer", "company": "TestCo"}, "")
    output = jd_generator_brain._template_fallback(params)
    assert "DevOps Engineer" in output
    assert "TestCo" in output


# ─── ResumeParserBrain ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resume_parser_empty():
    state = BrainState(query="")
    result = await resume_parser_brain.run(state)
    assert result.error == "No resume content provided"


@pytest.mark.asyncio
async def test_resume_parser_llm_success(mock_ollama):
    state = BrainState(query="John Doe\njohn@email.com\nPython developer with 5 years experience")
    result = await resume_parser_brain.run(state)
    assert result.result is not None
    assert result.metadata.get("parser") == "llm"


@pytest.mark.asyncio
async def test_resume_parser_fallback(mock_ollama_failure):
    state = BrainState(query="John Doe\njohn@email.com\nPython developer skilled in Django, React")
    result = await resume_parser_brain.run(state)
    assert result.result is not None
    assert result.metadata.get("parser") == "fallback"
    assert "personal_info" in result.result
    assert "skills" in result.result


@pytest.mark.asyncio
async def test_resume_parser_extract_email():
    info = resume_parser_brain._extract_personal_info("Contact me at john@example.com")
    assert info["email"] == "john@example.com"


@pytest.mark.asyncio
async def test_resume_parser_extract_phone():
    info = resume_parser_brain._extract_personal_info("Phone: +1 (555) 123-4567")
    assert info["phone"] != ""


@pytest.mark.asyncio
async def test_resume_parser_extract_linkedin():
    info = resume_parser_brain._extract_personal_info("linkedin.com/in/johndoe")
    assert info["linkedin"] == "linkedin.com/in/johndoe"


@pytest.mark.asyncio
async def test_resume_parser_extract_skills():
    from unittest.mock import MagicMock
    skills = resume_parser_brain._extract_skills("Python, JavaScript, Docker, Django")
    assert "python" in [s.lower() for s in skills["technical"]]
    assert "django" in [s.lower() for s in skills["frameworks"]]


# ─── ATSBrain ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ats_brain_no_resume():
    state = BrainState(query="")
    result = await ats_brain.run(state)
    assert result.error == "No resume provided"


@pytest.mark.asyncio
async def test_ats_brain_rule_based():
    state = BrainState(
        query="Python developer with React experience",
        context={"job_description": "Looking for Python, React, Docker expert"},
    )
    result = await ats_brain.run(state)
    assert result.result is not None
    assert "ats_score" in result.result
    assert result.result["keyword_match"]["matched"] is not None


@pytest.mark.asyncio
async def test_ats_brain_fallback(mock_ollama_failure):
    state = BrainState(
        query="Python and React developer",
        context={"job_description": "Need Python, Docker, AWS skills"},
    )
    result = await ats_brain.run(state)
    assert result.result is not None
    assert result.metadata.get("model") == "rule_based"
    assert "ats_score" in result.result


@pytest.mark.asyncio
async def test_ats_brain_empty_result():
    result = ats_brain._empty_result()
    assert result["ats_score"] == 0
    assert result["passes_ats"] is False


# ─── JobMatchingBrain ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_matching_missing_fields():
    state = BrainState(query="")
    result = await job_matching_brain.run(state)
    assert "Both candidate profile and job requirements required" in result.error


@pytest.mark.asyncio
async def test_job_matching_fallback(mock_ollama_failure):
    state = BrainState(
        query="Python developer with 3 years experience",
        context={"job_requirements": "Need Python, React, Docker. 2+ years experience."},
    )
    result = await job_matching_brain.run(state)
    assert result.result is not None
    assert result.metadata.get("model") == "rule_based"
    assert "match_score" in result.result
    assert "recommendation" in result.result


@pytest.mark.asyncio
async def test_job_matching_rule_based_high_match():
    state = BrainState(
        query="Python, React, Docker, AWS developer with 5 years",
        context={"job_requirements": "Python, React, Docker, AWS required. 3+ years."},
    )
    result = await job_matching_brain.run(state)
    assert result.result["match_score"] >= 60
    assert result.result["recommendation"] in ["strong_match", "good_match"]


@pytest.mark.asyncio
async def test_job_matching_rule_based_low_match():
    state = BrainState(
        query="Java developer",
        context={"job_requirements": "Python, React, Docker required. 3+ years."},
    )
    result = await job_matching_brain.run(state)
    assert result.result["match_score"] < 60


# ─── CareerBrain ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_career_advice_fallback(mock_ollama_failure):
    state = BrainState(
        query="Career advice",
        intent="CAREER_ADVICE",
        context={"current_role": "Junior Dev", "target_role": "Senior Dev"},
    )
    result = await career_brain.run(state)
    assert result.result is not None
    assert "career_path" in result.result
    assert result.metadata.get("fallback") is True


@pytest.mark.asyncio
async def test_skill_assessment_fallback(mock_ollama_failure):
    state = BrainState(
        query="Test my Python skills",
        intent="SKILL_ASSESSMENT",
        context={"skill": "Python", "level": "intermediate"},
    )
    result = await career_brain.run(state)
    assert result.result is not None
    assert "questions" in result.result
    assert result.result["questions"] == []


@pytest.mark.asyncio
async def test_interview_prep_fallback(mock_ollama_failure):
    state = BrainState(
        query="Interview prep for React dev",
        intent="INTERVIEW_PREP",
        context={"role": "React Developer", "level": "mid"},
    )
    result = await career_brain.run(state)
    assert result.result is not None
    assert "questions" in result.result
    assert result.result["topics_to_review"] == []


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
    assert result.result is not None
    assert "summary" in result.result


@pytest.mark.asyncio
async def test_career_unknown_intent_defaults_to_advice(mock_ollama_failure):
    state = BrainState(query="Something", intent="UNKNOWN")
    result = await career_brain.run(state)
    assert result.result is not None


# ─── RecruiterBrain ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recruiter_search_fallback(mock_ollama_failure):
    state = BrainState(
        query="Find Python developers",
        user_id="emp_user",
        user_role="employer",
        context={"skills": ["Python", "Django"]},
    )
    result = await recruiter_brain.run(state)
    assert result.result is not None
    assert result.metadata.get("fallback") is True
    assert "search_strategy" in result.result
    assert "Python" in result.result["search_strategy"]


@pytest.mark.asyncio
async def test_recruiter_shortlist_fallback(mock_ollama_failure):
    state = BrainState(
        query="Shortlist the best candidates",
        context={
            "job_requirements": "Python developer",
            "candidates": [{"name": "Alice"}, {"name": "Bob"}],
        },
    )
    result = await recruiter_brain.run(state)
    assert result.result is not None
    assert "shortlisted" in result.result


@pytest.mark.asyncio
async def test_recruiter_routes_to_search_by_default():
    state = BrainState(query="Find candidates", context={"skills": ["Python"]})
    result = await recruiter_brain.run(state)
    # Should return results without error
    assert result.error is None or "error" not in (result.error or "").lower()
    assert result.result is not None
    assert "search_strategy" in result.result or "recommended_filters" in result.result


@pytest.mark.asyncio
async def test_recruiter_shortlist_route(mock_ollama_failure):
    state = BrainState(
        query="shortlist candidates",
        context={"job_requirements": "Python dev", "candidates": [{"name": "Alice"}]},
    )
    result = await recruiter_brain.run(state)
    assert result.error is None or result.result is not None

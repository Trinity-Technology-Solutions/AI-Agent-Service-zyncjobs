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


# ─── ATS Resume Analyzer — Comprehensive Test Cases (ZP-461 to ZP-480) ──────

@pytest.mark.asyncio
async def test_zp461_ats_score_for_valid_resume_and_jd():
    """ZP-461: Verify ATS score is generated for a valid resume and Job Description"""
    state = BrainState(
        query="John Doe\nPython developer with 5 years experience in React, Django, PostgreSQL\nSkills: Python, React, Django, PostgreSQL, Docker, AWS\nExperience:\n- Senior Python Developer at TechCorp (2020-2024)\n  • Built scalable APIs using Django REST Framework\n  • Deployed applications on AWS using Docker\nEducation:\n- B.Tech Computer Science, IIT Delhi (2016-2020)",
        context={"job_description": "We are looking for a Senior Python Developer with experience in React, Django, PostgreSQL, Docker, and AWS. 3+ years experience required."},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert "ats_score" in result.response
    assert isinstance(result.response["ats_score"], (int, float))
    assert 0 <= result.response["ats_score"] <= 100
    assert "keyword_match" in result.response
    assert "formatting_score" in result.response
    assert "section_completeness" in result.response
    assert "experience_relevance" in result.response
    assert "suggestions" in result.response
    assert "passes_ats" in result.response


@pytest.mark.asyncio
async def test_zp462_matched_keywords_highlighted():
    """ZP-462: Verify matched keywords are highlighted"""
    state = BrainState(
        query="Python developer with React, Django, PostgreSQL skills",
        context={"job_description": "Need Python, React, Django expert"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    km = result.response.get("keyword_match", {})
    matched = km.get("matched", [])
    assert "python" in [m.lower() for m in matched]
    assert "react" in [m.lower() for m in matched]
    assert "django" in [m.lower() for m in matched]


@pytest.mark.asyncio
async def test_zp463_missing_keywords_displayed():
    """ZP-463: Verify missing keywords are displayed"""
    state = BrainState(
        query="Python developer with React skills",
        context={"job_description": "Need Python, React, Docker, AWS, Kubernetes expert"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    km = result.response.get("keyword_match", {})
    missing = km.get("missing", [])
    # Should detect missing keywords
    missing_lower = [m.lower() for m in missing]
    assert any("docker" in m for m in missing_lower)
    assert any("aws" in m for m in missing_lower) or any("kubernetes" in m for m in missing_lower)


@pytest.mark.asyncio
async def test_zp464_ai_displays_suggestions():
    """ZP-464: Verify AI displays resume improvement suggestions"""
    state = BrainState(
        query="Basic Python developer",
        context={"job_description": "Need Python, React, Docker, AWS, Kubernetes expert with 5+ years"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    suggestions = result.response.get("suggestions", [])
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    # Should contain actionable suggestions
    suggestion_text = " ".join(suggestions).lower()
    assert any(keyword in suggestion_text for keyword in ["add", "missing", "quantify", "expand", "section"])


@pytest.mark.asyncio
async def test_zp465_high_ats_score_for_matching_resume():
    """ZP-465: Verify high ATS score for highly matching resume"""
    state = BrainState(
        query="Senior Python Developer with 5 years experience in React, Django, PostgreSQL, Docker, AWS, Kubernetes\nSkills: Python, React, Django, PostgreSQL, Docker, AWS, Kubernetes, Git, CI/CD, REST APIs\nExperience:\n- Senior Developer at TechCorp (2020-2024)\n  • Built scalable microservices using Django and React\n  • Deployed on AWS using Docker and Kubernetes\n  • Implemented CI/CD pipelines\nEducation:\n- M.Tech Computer Science, IIT Bombay (2018-2020)",
        context={"job_description": "Senior Python Developer required. Must have React, Django, PostgreSQL, Docker, AWS, Kubernetes. 3+ years experience. CI/CD experience preferred."},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert result.response["ats_score"] >= 70  # Should pass ATS
    assert result.response["passes_ats"] is True


@pytest.mark.asyncio
async def test_zp466_low_ats_score_for_poorly_matching_resume():
    """ZP-466: Verify low ATS score for poorly matching resume"""
    state = BrainState(
        query="Java developer with Spring Boot experience",
        context={"job_description": "Senior Python Developer required. Must have React, Django, PostgreSQL, Docker, AWS. 5+ years Python experience."},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert result.response["ats_score"] < 60  # Should fail ATS
    assert result.response["passes_ats"] is False


@pytest.mark.asyncio
async def test_zp467_ats_scoring_with_empty_jd():
    """ZP-467: Verify ATS scoring with an empty Job Description"""
    state = BrainState(
        query="Python developer with React experience",
        context={"job_description": ""},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    # Should return error for empty JD
    assert "error" in result.response or result.response.get("ats_score", 100) == 0


@pytest.mark.asyncio
async def test_zp468_ats_scoring_with_long_jd():
    """ZP-468: Verify ATS scoring with a very long Job Description"""
    long_jd = "We are looking for a Senior Python Developer. " * 100 + "Skills: Python, React, Django, PostgreSQL, Docker, AWS, Kubernetes, Git, CI/CD, REST APIs, GraphQL, gRPC, Kafka, Redis, MongoDB, Terraform, Ansible, Jenkins, GitHub Actions."
    state = BrainState(
        query="Python developer with React, Django, PostgreSQL, Docker, AWS skills",
        context={"job_description": long_jd},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert "ats_score" in result.response
    assert 0 <= result.response["ats_score"] <= 100


@pytest.mark.asyncio
async def test_zp469_ats_scoring_with_multilingual_jd():
    """ZP-469: Verify ATS scoring with a multilingual Job Description"""
    multilingual_jd = "We need a Senior Python Developer. Required skills: Python, React, Django. Nous recherchons un développeur Python senior. Benötigte Fähigkeiten: Python, React, Django. 必要なスキル: Python, React, Django."
    state = BrainState(
        query="Python developer with React, Django skills",
        context={"job_description": multilingual_jd},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert "ats_score" in result.response
    assert 0 <= result.response["ats_score"] <= 100
    # Should still extract English keywords
    km = result.response.get("keyword_match", {})
    matched = [m.lower() for m in km.get("matched", [])]
    assert "python" in matched


@pytest.mark.asyncio
async def test_zp470_ats_score_updates_after_editing_resume():
    """ZP-470: Verify ATS score updates after editing resume"""
    # Initial resume - low match
    state1 = BrainState(
        query="Java developer",
        context={"job_description": "Python, React, Docker required"},
    )
    result1 = await ats_brain.run(state1)
    score1 = result1.response.get("ats_score", 0)
    
    # Updated resume - high match
    state2 = BrainState(
        query="Python developer with React, Docker, AWS skills",
        context={"job_description": "Python, React, Docker required"},
    )
    result2 = await ats_brain.run(state2)
    score2 = result2.response.get("ats_score", 0)
    
    assert score2 > score1


@pytest.mark.asyncio
async def test_zp471_ats_score_updates_after_changing_jd():
    """ZP-471: Verify ATS score updates after changing Job Description"""
    resume = "Python developer with React, Django, PostgreSQL skills"
    
    # JD matching skills
    state1 = BrainState(
        query=resume,
        context={"job_description": "Python, React, Django required"},
    )
    result1 = await ats_brain.run(state1)
    score1 = result1.response.get("ats_score", 0)
    
    # JD not matching
    state2 = BrainState(
        query=resume,
        context={"job_description": "Java, Spring, Hibernate required"},
    )
    result2 = await ats_brain.run(state2)
    score2 = result2.response.get("ats_score", 0)
    
    assert score1 > score2


@pytest.mark.asyncio
async def test_zp472_duplicate_keywords_handled():
    """ZP-472: Verify duplicate keywords are handled correctly"""
    state = BrainState(
        query="Python, Python, React, React, Django, Django developer",
        context={"job_description": "Python, React, Django required"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    km = result.response.get("keyword_match", {})
    matched = km.get("matched", [])
    # Should not have duplicates in matched
    matched_lower = [m.lower() for m in matched]
    assert len(matched_lower) == len(set(matched_lower))


@pytest.mark.asyncio
async def test_zp473_technical_skills_detected():
    """ZP-473: Verify technical skills are detected correctly"""
    state = BrainState(
        query="Full stack developer with Python, JavaScript, React, Node.js, PostgreSQL, MongoDB, Docker, Kubernetes, AWS, Git, REST APIs, GraphQL",
        context={"job_description": "Need Python, React, Node.js, PostgreSQL, Docker, AWS"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    km = result.response.get("keyword_match", {})
    matched = [m.lower() for m in km.get("matched", [])]
    # Technical skills should be detected
    tech_skills = ["python", "react", "node", "postgresql", "docker", "aws"]
    for skill in tech_skills:
        assert any(skill in m for m in matched), f"Technical skill '{skill}' not detected"


@pytest.mark.asyncio
async def test_zp474_soft_skills_detected(mock_ollama_failure):
    """ZP-474: Verify soft skills are detected correctly"""
    state = BrainState(
        query="Team player with excellent communication skills, leadership experience, problem solving, agile methodology, adaptable",
        context={"job_description": "Need communication, leadership, teamwork, agile, problem solving"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    km = result.response.get("keyword_match", {})
    matched = [m.lower() for m in km.get("matched", [])]
    soft_skills = ["communication", "leadership", "teamwork", "agile", "problem solving"]
    for skill in soft_skills:
        assert any(skill in m for m in matched), f"Soft skill '{skill}' not detected"


@pytest.mark.asyncio
async def test_zp475_experience_relevance_contributes_to_score():
    """ZP-475: Verify experience relevance contributes to ATS score"""
    # Resume with many skills (high experience relevance)
    state1 = BrainState(
        query="Senior developer with Python, React, Django, PostgreSQL, Docker, AWS, Kubernetes, Git, CI/CD, REST APIs, GraphQL, Kafka, Redis",
        context={"job_description": "Python, React, Docker required"},
    )
    result1 = await ats_brain.run(state1)
    score1 = result1.response.get("ats_score", 0)
    exp_rel1 = result1.response.get("experience_relevance", 0)
    
    # Resume with few skills (low experience relevance)
    state2 = BrainState(
        query="Junior developer with Python",
        context={"job_description": "Python, React, Docker required"},
    )
    result2 = await ats_brain.run(state2)
    score2 = result2.response.get("ats_score", 0)
    exp_rel2 = result2.response.get("experience_relevance", 0)
    
    assert exp_rel1 > exp_rel2
    assert score1 >= score2


@pytest.mark.asyncio
async def test_zp476_resume_formatting_impacts_score():
    """ZP-476: Verify resume formatting impacts ATS score"""
    # Well-formatted resume — has all 4 section headers: experience, education, skills, summary
    state1 = BrainState(
        query="Professional Summary\nSenior Python developer with 5 years experience.\nExperience\nSenior Developer at TechCorp (2020-2024)\n• Built scalable APIs\nEducation\nB.Tech Computer Science, IIT Delhi (2016-2020)\nSkills\nPython, React, Docker, AWS",
        context={"job_description": "Python, React required"},
    )
    result1 = await ats_brain.run(state1)
    formatting1 = result1.response.get("formatting_score", 0)

    # Poorly formatted resume — no section headers, just a single line
    state2 = BrainState(
        query="developer",
        context={"job_description": "Python, React required"},
    )
    result2 = await ats_brain.run(state2)
    formatting2 = result2.response.get("formatting_score", 0)

    assert formatting1 > formatting2


@pytest.mark.asyncio
async def test_zp477_incomplete_resume_receives_lower_score():
    """ZP-477: Verify incomplete resume receives lower score"""
    # Incomplete resume
    state1 = BrainState(
        query="Python",
        context={"job_description": "Python, React, Docker, AWS required"},
    )
    result1 = await ats_brain.run(state1)
    score1 = result1.response.get("ats_score", 0)
    completeness1 = result1.response.get("section_completeness", 0)
    
    # Complete resume
    state2 = BrainState(
        query="John Doe\nPython developer with 5 years\nEXPERIENCE\n• Built APIs\nEDUCATION\nB.Tech CS\nSKILLS\nPython, React, Docker, AWS",
        context={"job_description": "Python, React, Docker, AWS required"},
    )
    result2 = await ats_brain.run(state2)
    score2 = result2.response.get("ats_score", 0)
    completeness2 = result2.response.get("section_completeness", 0)
    
    assert completeness2 > completeness1
    assert score2 > score1


@pytest.mark.asyncio
async def test_zp478_ats_donut_chart_data():
    """ZP-478: Verify ATS donut chart displays correctly (data structure)"""
    state = BrainState(
        query="Python developer with React, Docker skills",
        context={"job_description": "Python, React, Docker, AWS required"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    # Check all required fields for chart display
    assert "ats_score" in result.response
    assert "keyword_match" in result.response
    assert "formatting_score" in result.response
    assert "section_completeness" in result.response
    assert "experience_relevance" in result.response
    assert isinstance(result.response["ats_score"], (int, float))
    assert 0 <= result.response["ats_score"] <= 100


@pytest.mark.asyncio
async def test_zp479_keyword_match_section_displayed():
    """ZP-479: Verify keyword match section is displayed correctly"""
    state = BrainState(
        query="Python developer with React, Docker skills",
        context={"job_description": "Python, React, Docker, AWS, Kubernetes required"},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    km = result.response.get("keyword_match", {})
    assert "matched" in km
    assert "missing" in km
    assert "match_percentage" in km
    assert isinstance(km["matched"], list)
    assert isinstance(km["missing"], list)
    assert isinstance(km["match_percentage"], (int, float))
    assert 0 <= km["match_percentage"] <= 100


@pytest.mark.asyncio
async def test_zp480_ats_scoring_works_after_deployment():
    """ZP-480: Verify ATS scoring works after new application deployment (regression test)"""
    # This test ensures the ATS scoring pipeline works end-to-end
    state = BrainState(
        query="John Doe\nSenior Python Developer\n5 years experience in React, Django, PostgreSQL, Docker, AWS\nEXPERIENCE\n• Built microservices with Django\n• Deployed on AWS with Docker\nEDUCATION\nM.Tech CS, IIT Bombay\nSKILLS\nPython, React, Django, PostgreSQL, Docker, AWS, Kubernetes, Git",
        context={"job_description": "Senior Python Developer. React, Django, PostgreSQL, Docker, AWS required. 3+ years."},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert "ats_score" in result.response
    assert result.response["ats_score"] >= 70  # Should pass
    assert result.response["passes_ats"] is True
    assert "keyword_match" in result.response
    assert "suggestions" in result.response
    assert len(result.response["suggestions"]) > 0


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

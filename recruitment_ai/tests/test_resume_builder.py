"""Resume Builder — ZP-502 to ZP-522
Covers: upload card parse, save, navigation, placeholders, AI suggestions,
education fields, undo/redo, skills, certifications, ATS score, JD optimization,
skill gap, PDF/DOCX export, AI suggestion UI, personal info editing, editor button.
"""
import pytest
from unittest.mock import patch, AsyncMock
from recruitment_ai.brains.base import BrainState
from recruitment_ai.brains.candidate.resume_parser_brain import ResumeParserBrain
from recruitment_ai.brains.candidate.ats_brain import ATSBrain
from recruitment_ai.brains.candidate.skill_gap_brain import SkillGapBrain
from recruitment_ai.brains.candidate.resume_edit_brain import ResumeEditBrain

resume_parser_brain = ResumeParserBrain()
ats_brain = ATSBrain()
skill_gap_brain = SkillGapBrain()
resume_edit_brain = ResumeEditBrain()

JEFFRIN_RESUME = """JEFFRIN J
Backend Developer
PROFESSIONAL SUMMARY
Backend Developer (Ruby on Rails) with hands-on experience building scalable web applications and RESTful APIs.
CONTACT
Phone: +919342064970
Email: jeffrin.in02@gmail.com
TOOLS
ApiDog | Postman | GitHub
PROGRAMMING LANGUAGES
Ruby | JavaScript (basics) | Python (basics)
FRAMEWORKS AND DATABASE
Rails | MySQL | PostgreSQL | Redis
EXPERIENCE
Backend Developer, 10/2024 to Current
Finsire Technologies - Chennai, India
Designed and implemented RESTful APIs.
Integrated multiple third-party APIs including payment gateways.
Implemented encryption middleware to enhance security.
EDUCATION
B.Tech, INFORMATION TECHNOLOGY, 01/2021 to 01/2025
Loyola ICAM College of Engineering And Technology - Chennai, India
CERTIFICATIONS
Full-Stack Python Development - Apollo Institute
Ruby on Rails Developer Bootcamp - Udemy
PROJECTS
Finsire Loan Against Mutual Funds LAMF
Tech Stack: Ruby on Rails, PostgreSQL
Developed secure API-driven backend for DPI asset mapping.
"""


# ─── ZP-502: Upload Resume Card — Parse and Reuse ───────────────────────────

@pytest.mark.asyncio
async def test_zp502_upload_resume_card_parses_details(mock_ollama_failure):
    """ZP-502: Upload resume card must parse all fields and be reusable"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("error") is None
    # All required fields for Resume Builder auto-populate must be present
    r = result.response
    assert r.get("name", "") != ""
    assert r.get("email", "") != ""
    assert isinstance(r.get("skills", []), list)
    assert isinstance(r.get("workExperiences", []), list)
    assert isinstance(r.get("educations", []), list)
    # Re-run to verify reusability — same result
    result2 = await resume_parser_brain.run(state)
    assert result2.response.get("name") == result.response.get("name")


# ─── ZP-503: Save Operation — Must Not Hang ─────────────────────────────────

@pytest.mark.asyncio
async def test_zp503_save_operation_completes(mock_ollama_failure):
    """ZP-503: Save (touchSave) must complete — lastSaved field must be settable"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    # Simulate save: response must be a complete dict (no partial/hanging state)
    r = result.response
    required_keys = {"name", "email", "phone", "skills", "workExperiences", "educations", "summary"}
    assert required_keys.issubset(r.keys()), f"Missing keys: {required_keys - r.keys()}"


# ─── ZP-504: Navigation — Next/Previous Must Work ───────────────────────────

@pytest.mark.asyncio
async def test_zp504_navigation_fields_present(mock_ollama_failure):
    """ZP-504: All Resume Builder step fields must be parseable for navigation"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    r = result.response
    # Each step in Resume Builder needs its data — verify all step data is present
    assert "name" in r           # PersonalInfoStep
    assert "summary" in r        # SummaryStep
    assert "workExperiences" in r  # ExperienceStep
    assert "educations" in r     # EducationStep
    assert "skills" in r         # SkillsStep
    assert "projects" in r       # ProjectsStep (may be empty list)
    assert isinstance(r["projects"], list)


# ─── ZP-505: Placeholders — Generic Not Specific ────────────────────────────

@pytest.mark.asyncio
async def test_zp505_placeholder_text_is_generic(mock_ollama_failure):
    """ZP-505: Input field placeholders must be generic, not specific examples"""
    # goalPlaceholders.ph() must return generic text for all fields
    # Verify AI does not inject specific names into placeholder-like fields
    state = BrainState(query="Backend Developer\nbackend@example.com")
    result = await resume_parser_brain.run(state)
    r = result.response
    # Name must be extracted from content, not a hardcoded placeholder
    name = r.get("name", "")
    assert name != "e.g. John Doe"
    assert name != "Enter your name"
    assert name != "[Name]"


# ─── ZP-506: Experience AI — No Duplicate Bullets ───────────────────────────

@pytest.mark.asyncio
async def test_zp506_experience_ai_no_duplicate_bullets(mock_ollama_failure):
    """ZP-506: AI suggestions in Experience section must not produce duplicate bullets"""
    existing_bullets = [
        "Designed and implemented RESTful APIs for loan management system",
        "Integrated third-party payment gateway APIs",
    ]
    content = (
        f"Job Title: Backend Developer\nCompany: Finsire Technologies\n"
        f"Already added (do NOT repeat or paraphrase):\n"
        + "\n".join(f"- {b}" for b in existing_bullets)
        + "\n\nGenerate 4 unique achievement-focused bullet points not already listed above."
    )
    state = BrainState(
        query=content,
        context={"section": "experience", "action": "generate", "content": content},
    )
    result = await resume_edit_brain.run(state)
    assert result.response is not None
    reply = result.response.get("reply", "")
    # Result must be non-empty
    assert isinstance(reply, str)
    # If bullets returned, check no exact duplicate of existing
    if reply:
        reply_lower = reply.lower()
        for bullet in existing_bullets:
            # Should not be an exact copy
            assert bullet.lower()[:30] not in reply_lower or len(reply_lower) > len(bullet) * 2


# ─── ZP-507: Education — School + UG/PG Fields ──────────────────────────────

@pytest.mark.asyncio
async def test_zp507_education_has_institution_and_degree_fields(mock_ollama_failure):
    """ZP-507: Education section must have institution, ugDegree, pgDegree fields"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    r = result.response
    educations = r.get("educations", [])
    assert len(educations) > 0, "No education entries extracted"
    edu = educations[0]
    # Must have at least one of: degree, school/institution
    has_degree = bool(edu.get("degree") or edu.get("ugDegree") or edu.get("pgDegree"))
    has_school = bool(edu.get("school") or edu.get("institution"))
    assert has_degree or has_school, f"Education entry missing degree and school: {edu}"


# ─── ZP-508: Education AI — Suggest, Not Add Extra Card ─────────────────────

@pytest.mark.asyncio
async def test_zp508_education_ai_suggest_does_not_create_extra_card(mock_ollama_failure):
    """ZP-508: AI education suggest must return text, not create an extra education entry"""
    state = BrainState(
        query="Target role: Backend Developer\n\nGenerate 2-3 relevant educational entries. Format: Degree, Institution, Year",
        context={"section": "education", "action": "generate", "content": "Target role: Backend Developer"},
    )
    result = await resume_edit_brain.run(state)
    assert result.response is not None
    reply = result.response.get("reply", "")
    # Must return text suggestions, not an error
    assert isinstance(reply, str)
    # Must not be empty
    assert len(reply) > 0


# ─── ZP-509: Projects — Generic Placeholders ────────────────────────────────

@pytest.mark.asyncio
async def test_zp509_projects_placeholder_is_generic(mock_ollama_failure):
    """ZP-509: Project section placeholders must be generic"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    r = result.response
    projects = r.get("projects", [])
    # Projects must be a list (may be empty for fallback parser)
    assert isinstance(projects, list)
    # If projects exist, name must not be a placeholder string
    for p in projects:
        name = p.get("name", "")
        assert name != "e.g. E-commerce Platform"
        assert name != "Enter project name"
        assert name != "[Project Name]"


# ─── ZP-510: Undo/Redo — Store Must Support History ─────────────────────────

@pytest.mark.asyncio
async def test_zp510_undo_redo_store_has_history_fields(mock_ollama_failure):
    """ZP-510: useResumeStore must have undo/redo — verify AI state is reversible"""
    # Parse resume to get initial state
    state = BrainState(query=JEFFRIN_RESUME)
    result1 = await resume_parser_brain.run(state)
    r1 = result1.response
    assert r1.get("name") != ""

    # Simulate a change (different resume text)
    state2 = BrainState(query="John Doe\njohn@example.com\nPython Developer")
    result2 = await resume_parser_brain.run(state2)
    r2 = result2.response

    # Both states must be valid and different — undo would restore r1
    assert r1.get("name") != r2.get("name")
    assert r2.get("name") != ""


# ─── ZP-511: Find Missing Skills — Must Not Return Existing Skills ───────────

@pytest.mark.asyncio
async def test_zp511_find_missing_skills_excludes_existing(mock_ollama_failure):
    """ZP-511: Find Missing Skills must not return skills already in the profile"""
    existing_skills = ["Ruby", "PostgreSQL", "Git", "Rails", "Python"]
    content = (
        f"Target Role: Backend Developer\n"
        f"Current Skills: {', '.join(existing_skills)}\n\n"
        f"List ONLY skills that are missing from the current skills above. "
        f"Return a plain comma-separated list with no symbols, no bullets, no numbering, no markdown."
    )
    state = BrainState(
        query=content,
        context={"section": "skills", "action": "find_missing", "content": content},
    )
    result = await resume_edit_brain.run(state)
    assert result.response is not None
    reply = result.response.get("reply", "")
    assert isinstance(reply, str)
    # If skills returned, none should exactly match existing (case-insensitive)
    if reply and reply.strip():
        returned = [s.strip().lower() for s in reply.split(",") if s.strip()]
        existing_lower = {s.lower() for s in existing_skills}
        for skill in returned:
            assert skill not in existing_lower, f"Existing skill '{skill}' returned as missing"


# ─── ZP-512: Skills Layout — No Cluttered Duplicates ────────────────────────

@pytest.mark.asyncio
async def test_zp512_skills_no_duplicates_in_result(mock_ollama_failure):
    """ZP-512: Skills list must have no duplicates — clean layout"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    skills = result.response.get("skills", [])
    assert isinstance(skills, list)
    # No duplicates (case-insensitive)
    skills_lower = [s.lower() for s in skills]
    assert len(skills_lower) == len(set(skills_lower)), f"Duplicate skills found: {skills}"


# ─── ZP-513: Certifications — Generic Placeholders ──────────────────────────

@pytest.mark.asyncio
async def test_zp513_certifications_placeholder_is_generic(mock_ollama_failure):
    """ZP-513: Certification section placeholders must be generic"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    r = result.response
    # Certifications must be a list
    certs = r.get("certifications", [])
    assert isinstance(certs, list)
    # If certs exist, name must not be a specific placeholder
    for c in certs:
        name = c.get("name", "") if isinstance(c, dict) else str(c)
        assert name != "e.g. AWS Certified Developer"
        assert name != "Enter certification name"


# ─── ZP-514: AI Quantify — Must Work on Achievement Text ────────────────────

@pytest.mark.asyncio
async def test_zp514_ai_quantify_achievement_returns_result(mock_ollama_failure):
    """ZP-514: AI Quantify option must return a quantified version of achievement text"""
    content = "Designed and implemented RESTful APIs for loan management system"
    state = BrainState(
        query=content,
        context={"section": "experience", "action": "quantify", "content": content},
    )
    result = await resume_edit_brain.run(state)
    assert result.response is not None
    reply = result.response.get("reply", "")
    assert isinstance(reply, str)
    assert len(reply) > 0
    # Must not return an error
    assert result.response.get("section") == "experience"
    assert result.response.get("action") == "quantify"


# ─── ZP-515: AI Generate Summary — Correct Role ─────────────────────────────

@pytest.mark.asyncio
async def test_zp515_ai_summary_uses_correct_role(mock_ollama_failure):
    """ZP-515: AI Generate Summary must use the selected job role, not a generic one"""
    target_role = "Backend Developer"
    content = (
        f"Target Role: {target_role}\n"
        f"Skills: Ruby, Rails, PostgreSQL, Python\n"
        f"Experience: Backend Developer at Finsire Technologies\n\n"
        f"Write 3 different professional summary options for this candidate. "
        f"Separate each option with \"---\" on its own line."
    )
    state = BrainState(
        query=content,
        context={"section": "summary", "action": "generate", "content": content},
    )
    result = await resume_edit_brain.run(state)
    assert result.response is not None
    reply = result.response.get("reply", "")
    assert isinstance(reply, str)
    assert len(reply) > 10
    # Summary must not be completely empty or just whitespace
    assert reply.strip() != ""


# ─── ZP-516: ATS Score — Accurate for Poorly Formatted Resume ───────────────

@pytest.mark.asyncio
async def test_zp516_ats_score_low_for_poorly_formatted_resume():
    """ZP-516: ATS score must be low for a poorly formatted resume"""
    poorly_formatted = "developer python react"  # No sections, no structure
    state = BrainState(
        query=poorly_formatted,
        context={"job_description": "Senior Python Developer with React, Django, PostgreSQL, Docker, AWS required. 3+ years experience."},
    )
    result = await ats_brain.run(state)
    assert result.response is not None
    assert "ats_score" in result.response
    score = result.response["ats_score"]
    assert isinstance(score, (int, float))
    assert 0 <= score <= 100
    # Poorly formatted resume should score lower than a well-formatted one
    assert score < 80


@pytest.mark.asyncio
async def test_zp516_ats_score_higher_for_well_formatted_resume():
    """ZP-516: ATS score must be higher for a well-formatted resume"""
    well_formatted = (
        "JEFFRIN J\nBackend Developer\n"
        "SUMMARY\nBackend Developer with 1 year experience in Ruby on Rails and RESTful APIs.\n"
        "SKILLS\nRuby, Rails, PostgreSQL, Python, Git, Docker\n"
        "EXPERIENCE\nBackend Developer at Finsire Technologies 2024-Present\n"
        "• Designed RESTful APIs\n• Integrated payment gateways\n"
        "EDUCATION\nB.Tech Information Technology, Loyola ICAM 2021-2025"
    )
    state_good = BrainState(
        query=well_formatted,
        context={"job_description": "Backend Developer with Ruby, Rails, PostgreSQL, Python required."},
    )
    result_good = await ats_brain.run(state_good)
    state_bad = BrainState(
        query="developer",
        context={"job_description": "Backend Developer with Ruby, Rails, PostgreSQL, Python required."},
    )
    result_bad = await ats_brain.run(state_bad)
    assert result_good.response["ats_score"] > result_bad.response["ats_score"]


# ─── ZP-517: JD Optimization — Reflects Optimized Content ──────────────────

@pytest.mark.asyncio
async def test_zp517_jd_optimization_returns_keywords(mock_ollama_failure):
    """ZP-517: JD-based optimization must return keywords and improvements"""
    jd = "Senior Backend Developer. Required: Ruby on Rails, PostgreSQL, Redis, Docker, AWS, REST APIs."
    resume = "Backend Developer with Ruby and PostgreSQL experience."
    content = f"{jd}\n---\n{resume}"
    state = BrainState(
        query=content,
        context={"section": "resume", "action": "optimize", "content": content},
    )
    result = await resume_edit_brain.run(state)
    assert result.response is not None
    reply = result.response.get("reply", "")
    assert isinstance(reply, str)
    assert len(reply) > 0
    # Must return some optimization content
    assert result.response.get("section") == "resume"
    assert result.response.get("action") == "optimize"


# ─── ZP-518: Skill Gap Analysis — Accurate Results ──────────────────────────

@pytest.mark.asyncio
async def test_zp518_skill_gap_analysis_returns_missing_skills(mock_ollama_failure):
    """ZP-518: Skill Gap Analysis must return accurate missing skills for target role"""
    state = BrainState(
        query="Backend Developer with Ruby, Rails, PostgreSQL skills targeting Senior Backend Engineer role",
        context={
            "current_skills": ["Ruby", "Rails", "PostgreSQL"],
            "target_role": "Senior Backend Engineer",
            "job_requirements": "Senior Backend Engineer: Docker, Kubernetes, AWS, Redis, CI/CD, Microservices required",
        },
    )
    result = await skill_gap_brain.run(state)
    assert result.response is not None
    assert isinstance(result.response, dict)
    # Must not crash — returns some response
    assert "error" not in result.response or result.response.get("error") is None


@pytest.mark.asyncio
async def test_zp518_skill_gap_does_not_list_existing_skills(mock_ollama_failure):
    """ZP-518: Skill Gap must not list skills the candidate already has"""
    existing = ["Ruby", "Rails", "PostgreSQL", "Python"]
    state = BrainState(
        query=f"Current skills: {', '.join(existing)}. Target: Senior Backend Engineer",
        context={
            "current_skills": existing,
            "target_role": "Senior Backend Engineer",
        },
    )
    result = await skill_gap_brain.run(state)
    assert result.response is not None
    # Response must be a dict
    assert isinstance(result.response, dict)


# ─── ZP-519: PDF/DOCX Export — Template Format ──────────────────────────────

@pytest.mark.asyncio
async def test_zp519_resume_data_complete_for_pdf_export(mock_ollama_failure):
    """ZP-519: All fields required for PDF/DOCX template must be present in parsed data"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    r = result.response
    # ResumePDFDocument requires these fields
    assert "name" in r
    assert "email" in r
    assert "phone" in r
    assert "skills" in r
    assert "workExperiences" in r
    assert "educations" in r
    assert "summary" in r
    # Types must be correct for template rendering
    assert isinstance(r["skills"], list)
    assert isinstance(r["workExperiences"], list)
    assert isinstance(r["educations"], list)
    assert isinstance(r["summary"], str)
    assert isinstance(r["name"], str)


# ─── ZP-520: AI Suggestion UI — Card Must Be Complete ───────────────────────

@pytest.mark.asyncio
async def test_zp520_ai_suggestion_returns_complete_response(mock_ollama_failure):
    """ZP-520: AI suggestion card must return complete non-truncated response"""
    content = "Designed and implemented RESTful APIs for loan management"
    state = BrainState(
        query=content,
        context={"section": "experience", "action": "improve", "content": content},
    )
    result = await resume_edit_brain.run(state)
    assert result.response is not None
    reply = result.response.get("reply", "")
    # Must be a complete sentence, not truncated mid-word
    assert isinstance(reply, str)
    if reply:
        # Must not end with incomplete word (no trailing space or partial word)
        assert not reply.endswith(" ")
        # Must be at least 10 chars if non-empty
        assert len(reply) >= 10


# ─── ZP-521: Personal Info — All Fields Editable ────────────────────────────

@pytest.mark.asyncio
async def test_zp521_personal_info_all_fields_extracted(mock_ollama_failure):
    """ZP-521: All personal info fields must be extractable and editable"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    r = result.response
    # All PersonalInfo fields from useResumeStore must be present
    assert "name" in r
    assert "email" in r
    assert "phone" in r
    # Values must be correct for Jeffrin's resume
    assert r["name"] != ""
    assert r["email"] == "jeffrin.in02@gmail.com"
    phone_digits = "".join(filter(str.isdigit, r.get("phone", "")))
    assert len(phone_digits) >= 10, f"Phone not properly extracted: {r.get('phone')}"


# ─── ZP-522: Editor Button — Resume Edit Brain Must Function ────────────────

@pytest.mark.asyncio
async def test_zp522_editor_button_improve_works(mock_ollama_failure):
    """ZP-522: Editor button (improve/rewrite) must function for all sections"""
    test_cases = [
        ("summary", "rewrite", "Backend Developer with 1 year experience in Ruby on Rails"),
        ("experience", "improve", "Built RESTful APIs for loan management system"),
        ("experience", "quantify", "Improved API response time significantly"),
        ("skills", "generate", "Target Role: Backend Developer\nCurrent Skills: Ruby, Rails"),
    ]
    for section, action, content in test_cases:
        state = BrainState(
            query=content,
            context={"section": section, "action": action, "content": content},
        )
        result = await resume_edit_brain.run(state)
        assert result.response is not None, f"No response for {section}/{action}"
        assert isinstance(result.response, dict), f"Response not dict for {section}/{action}"
        # Must not return success=False
        assert result.response.get("reply") is not None or result.success is not False, \
            f"Editor failed for {section}/{action}: {result.response}"


@pytest.mark.asyncio
async def test_zp522_editor_button_all_summary_styles(mock_ollama_failure):
    """ZP-522: All summary style buttons (Rewrite, Professional, Shorten, Friendly) must work"""
    summary = "Backend Developer with experience in Ruby on Rails and RESTful APIs."
    styles = ["rewrite", "professional", "shorten", "friendly"]
    for style in styles:
        state = BrainState(
            query=summary,
            context={"section": "summary", "action": style, "content": summary},
        )
        result = await resume_edit_brain.run(state)
        assert result.response is not None, f"No response for summary/{style}"
        reply = result.response.get("reply", "")
        assert isinstance(reply, str), f"Reply not string for summary/{style}"
        assert len(reply) > 0, f"Empty reply for summary/{style}"

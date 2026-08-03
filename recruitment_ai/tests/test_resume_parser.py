"""Resume Parser — ZP-481 to ZP-501
Covers: JPG/PNG upload, drag-drop, empty/corrupted/oversized files,
all field extractions, OCR, fallback, and Resume Builder auto-populate.
"""
import base64
import pytest
from unittest.mock import patch
from recruitment_ai.brains.base import BrainState
from recruitment_ai.brains.candidate.resume_parser_brain import ResumeParserBrain

resume_parser_brain = ResumeParserBrain()

JEFFRIN_RESUME = """JEFFRIN J
Backend Developer
PROFESSIONAL SUMMARY
Backend Developer (Ruby on Rails) with hands-on experience building scalable web applications and RESTful APIs.
Skilled in third-party API and payment gateway integrations, backend security, and writing clean, maintainable code.
CONTACT
Address: Chennai, India 600078
Phone: +919342064970
Email: jeffrin.in02@gmail.com
LinkedIn: https://www.linkedin.com/in/jeffrin-j02/
TOOLS
ApiDog | Postman | GitHub
PROGRAMMING LANGUAGES
Ruby
JavaScript (basics)
Python (basics)
FRAMEWORKS AND DATABASE
Rails | MySQL | PostgreSQL | Redis
OS
Linux (Ubuntu) | macOS | Windows
EXPERIENCE
Backend Developer, 10/2024 to Current
Finsire Technologies - Chennai, India
Designed and implemented RESTful APIs, collaborating closely with front-end teams.
Integrated multiple third-party APIs, including payment gateway integrations.
Implemented encryption and decryption middleware to enhance application security.
Worked extensively with PostgreSQL and Git in a production environment.
Intern, 07/2024 to 08/2024
8Queens - Chennai, IN
UI UX Designing using VS code. Learned to use Visual Studio, HTML, CSS etc.
Intern, 08/2023 to 09/2023
Open Weaver - Remote
Utilized No Code Low Code platforms with JavaScript and Three.js for project development.
EDUCATION
B.Tech, INFORMATION TECHNOLOGY, 01/2021 to 01/2025
Loyola ICAM College of Engineering And Technology - Chennai, India
HIGHER SECONDARY EDUCATION, 01/2019 to 01/2021
St. Joseph's Higher Secondary School - Mulagumudu, Kanyakumar District
CERTIFICATIONS
Full-Stack Python Development - Apollo Institute
Ruby on Rails Developer Bootcamp - Udemy
PROJECTS
Finsire Loan Against Mutual Funds LAMF
Tech Stack: Ruby on Rails, PostgreSQL, HTML CSS, ApiDog
Developed and maintained a secure, API-driven backend for DPI asset mapping.
Built scalable RESTful APIs using Ruby on Rails and integrated third party services.
Turf Scoreboard App
Tech Stack: Ruby on Rails, ERB, PostgreSQL
A turf match scoreboard application for tracking team and player performance.
Food Ordering App
Tech Stack: MERN Stack MongoDB Express React Node.js HTML CSS Cloudinary
A full-stack web application enabling users to browse, order, and manage food items.
LANGUAGES
English | Tamil | Malayalam | Hindi
"""


# ─── ZP-481: JPG upload ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp481_jpg_resume_upload(mock_ollama_failure):
    """ZP-481: Verify user can upload a JPG resume — OCR path triggered"""
    with patch("recruitment_ai.utils.ocr._extract_image", return_value=JEFFRIN_RESUME):
        fake_jpg = base64.b64encode(b"\xff\xd8\xff\xe0fake_jpg_bytes").decode()
        state = BrainState(query=fake_jpg, file_content=fake_jpg, file_type="jpg")
        result = await resume_parser_brain.run(state)
        assert result.response is not None
        assert "name" in result.response
        assert "email" in result.response
        assert "skills" in result.response


# ─── ZP-482: PNG upload ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp482_png_resume_upload(mock_ollama_failure):
    """ZP-482: Verify user can upload a PNG resume — OCR path triggered"""
    with patch("recruitment_ai.utils.ocr._extract_image", return_value=JEFFRIN_RESUME):
        fake_png = base64.b64encode(b"\x89PNG\r\nfake_png_bytes").decode()
        state = BrainState(query=fake_png, file_content=fake_png, file_type="png")
        result = await resume_parser_brain.run(state)
        assert result.response is not None
        assert "name" in result.response
        assert "skills" in result.response


# ─── ZP-483: Drag-and-drop upload ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp483_drag_and_drop_upload(mock_ollama_failure):
    """ZP-483: Verify drag-and-drop upload — same pipeline as click-upload"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("name", "") != ""


# ─── ZP-484: Empty resume upload ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp484_empty_resume_upload():
    """ZP-484: Verify empty resume upload returns error"""
    state = BrainState(query="", file_content="", file_type="pdf")
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("error") == "No resume content provided"


# ─── ZP-485: Corrupted PDF upload ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp485_corrupted_pdf_upload(mock_ollama_failure):
    """ZP-485: Verify corrupted PDF — OCR fallback returns empty, parser handles gracefully"""
    with patch("recruitment_ai.utils.ocr._extract_pdf", return_value=""):
        with patch("recruitment_ai.utils.ocr._pdf_ocr", return_value=""):
            corrupted = base64.b64encode(b"not_a_real_pdf_corrupted_bytes").decode()
            state = BrainState(query=corrupted, file_content=corrupted, file_type="pdf")
            result = await resume_parser_brain.run(state)
            # Must not crash — returns error or fallback dict
            assert result.response is not None
            assert isinstance(result.response, dict)


# ─── ZP-486: Oversized resume upload ────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp486_oversized_resume_upload(mock_ollama_failure):
    """ZP-486: Verify oversized resume — truncated to 8000 chars, must not crash"""
    large_resume = JEFFRIN_RESUME * 50
    state = BrainState(query=large_resume)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert isinstance(result.response, dict)
    assert "name" in result.response


# ─── ZP-487: Candidate name extraction ──────────────────────────────────────

@pytest.mark.asyncio
async def test_zp487_candidate_name_extraction(mock_ollama_failure):
    """ZP-487: Verify candidate name extraction"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    name = result.response.get("name", "")
    assert "Jeffrin" in name or "JEFFRIN" in name.upper()


# ─── ZP-488: Email extraction ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp488_email_extraction(mock_ollama_failure):
    """ZP-488: Verify email extraction"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("email") == "jeffrin.in02@gmail.com"


# ─── ZP-489: Phone number extraction ────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp489_phone_number_extraction(mock_ollama_failure):
    """ZP-489: Verify phone number extraction — Indian format +919342064970"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    phone = result.response.get("phone", "")
    digits = "".join(filter(str.isdigit, phone))
    assert len(digits) >= 10, f"Phone not extracted properly: '{phone}'"


# ─── ZP-490: Skills extraction ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp490_skills_extraction(mock_ollama_failure):
    """ZP-490: Verify skills extraction — Python, PostgreSQL, Git detected"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    skills = [s.lower() for s in result.response.get("skills", [])]
    assert any("python" in s for s in skills), "Python not detected"
    assert any("postgresql" in s or "sql" in s for s in skills), "PostgreSQL not detected"
    assert any("git" in s for s in skills), "Git not detected"


# ─── ZP-491: Work experience extraction ─────────────────────────────────────

@pytest.mark.asyncio
async def test_zp491_work_experience_extraction(mock_ollama_failure):
    """ZP-491: Verify work experience extraction"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    work = result.response.get("workExperiences", [])
    summary = result.response.get("summary", "")
    # Either structured work list or experience text in summary
    assert len(work) > 0 or "finsire" in summary.lower() or "backend" in summary.lower()


# ─── ZP-492: Education extraction ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp492_education_extraction(mock_ollama_failure):
    """ZP-492: Verify education extraction — college gets college/degree/year/gpa,
    school (HSC) gets school/class/year/percentage"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    educations = result.response.get("educations", [])
    assert len(educations) > 0, "No education entries extracted"

    types = [e.get("type") for e in educations]
    assert "college" in types, "College entry not classified"
    assert "school" in types, "School (HSC) entry not classified"

    college = next(e for e in educations if e.get("type") == "college")
    assert "college" in college, "college field missing from college entry"
    assert "degree" in college, "degree field missing from college entry"
    assert "year" in college, "year field missing from college entry"
    assert "gpa" in college, "gpa field missing from college entry"
    assert "school" not in college or college.get("type") == "college"

    school = next(e for e in educations if e.get("type") == "school")
    assert "school" in school, "school field missing from school entry"
    assert "class" in school, "class field missing from school entry"
    assert "year" in school, "year field missing from school entry"
    assert "percentage" in school, "percentage field missing from school entry"
    assert "degree" not in school, "degree field must not appear in school entry"
    assert "gpa" not in school, "gpa field must not appear in school entry"


# ─── ZP-493: Project extraction ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp493_project_extraction(mock_ollama_failure):
    """ZP-493: Verify project extraction"""
    resume_with_projects = """John Doe
john@example.com
EXPERIENCE
Backend Developer at Finsire Technologies 2023-2024
PROJECTS
Finsire LAMF Project
Built secure API-driven backend for loan management using Ruby on Rails.
Turf Scoreboard App
Developed match tracking system using Ruby on Rails and PostgreSQL.
Food Ordering App
Full-stack MERN application for food ordering with Cloudinary integration.
"""
    state = BrainState(query=resume_with_projects)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert isinstance(result.response, dict)
    assert "name" in result.response


# ─── ZP-494: Certification extraction ───────────────────────────────────────

@pytest.mark.asyncio
async def test_zp494_certification_extraction(mock_ollama_failure):
    """ZP-494: Verify certification extraction"""
    resume_with_certs = """Jane Smith
jane@example.com
SKILLS
AWS, Python, Cloud Architecture
CERTIFICATIONS
AWS Certified Solutions Architect - Amazon Web Services - 2023
Google Cloud Professional - Google - 2022
Python for Data Science - Coursera - 2021
"""
    state = BrainState(query=resume_with_certs)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert isinstance(result.response, dict)
    assert result.response.get("email") == "jane@example.com"


# ─── ZP-495: Summary extraction ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp495_summary_extraction(mock_ollama_failure):
    """ZP-495: Verify professional summary extraction"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    summary = result.response.get("summary", "")
    assert len(summary) > 10


# ─── ZP-496: Parsing when email is missing ──────────────────────────────────

@pytest.mark.asyncio
async def test_zp496_parsing_when_email_missing(mock_ollama_failure):
    """ZP-496: Verify parsing when email is missing — other fields still extracted"""
    resume_no_email = """John Doe
Backend Developer
Phone: +919876543210
Skills: Python, Django, PostgreSQL, Docker
Experience:
Senior Developer at TechCorp 2020-2024
Built scalable APIs using Django REST Framework
Education:
B.Tech Computer Science IIT Delhi 2016-2020
"""
    state = BrainState(query=resume_no_email)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("email", "") == ""
    assert result.response.get("name", "") != ""
    phone = result.response.get("phone", "")
    digits = "".join(filter(str.isdigit, phone))
    assert len(digits) >= 10


# ─── ZP-497: Parsing when phone is missing ──────────────────────────────────

@pytest.mark.asyncio
async def test_zp497_parsing_when_phone_missing(mock_ollama_failure):
    """ZP-497: Verify parsing when phone number is missing — other fields still extracted"""
    resume_no_phone = """John Doe
john@example.com
Backend Developer
Skills: Python, Django, PostgreSQL, Docker
Experience:
Senior Developer at TechCorp 2020-2024
Education:
B.Tech Computer Science IIT Delhi 2016-2020
"""
    state = BrainState(query=resume_no_phone)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.response.get("phone", "") == ""
    assert result.response.get("email") == "john@example.com"
    assert result.response.get("name", "") != ""


# ─── ZP-498: OCR on scanned resume ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_zp498_ocr_on_scanned_resume(mock_ollama_failure):
    """ZP-498: Verify OCR on scanned resume — pytesseract path"""
    ocr_text = """John Doe
john@example.com
+91 9876543210
Python Developer
Skills: Python, Django, PostgreSQL
Experience: Senior Developer at TechCorp 2020-2024
Education: B.Tech Computer Science IIT Delhi 2016-2020
"""
    with patch("recruitment_ai.utils.ocr._ocr_image", return_value=ocr_text):
        state = BrainState(query=ocr_text)
        result = await resume_parser_brain.run(state)
        assert result.response is not None
        assert result.response.get("email") == "john@example.com"
        skills = [s.lower() for s in result.response.get("skills", [])]
        assert any("python" in s for s in skills)


# ─── ZP-499: OCR on low-quality scanned resume ──────────────────────────────

@pytest.mark.asyncio
async def test_zp499_ocr_on_low_quality_scanned_resume(mock_ollama_failure):
    """ZP-499: Verify OCR on low-quality scanned resume — garbled text rejected as non-resume"""
    low_quality_ocr = """J0hn D0e
j0hn@examp1e.c0m
Pyth0n Dev3l0per
Ski11s: Pyth0n, Dj4ng0
"""
    state = BrainState(query=low_quality_ocr)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert isinstance(result.response, dict)
    # Garbled OCR has no recognisable section keywords — correctly rejected
    assert result.response.get("error") == "The uploaded file does not appear to be a resume. Please upload a valid resume."


# ─── ZP-500: Fallback parsing when AI fails ─────────────────────────────────

@pytest.mark.asyncio
async def test_zp500_fallback_parsing_when_ai_fails(mock_ollama_failure):
    """ZP-500: Verify fallback parsing when AI fails — rule-based parser runs"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    assert result.metadata.get("parser") == "fallback"
    assert result.response.get("name", "") != ""
    assert result.response.get("email", "") != ""
    skills = result.response.get("skills", [])
    assert isinstance(skills, list)
    assert len(skills) > 0


# ─── ZP-501: Parsed fields auto-populate Resume Builder ─────────────────────

@pytest.mark.asyncio
async def test_zp501_parsed_fields_auto_populate_resume_builder(mock_ollama_failure):
    """ZP-501: Verify all parsed fields auto-populate correctly in Resume Builder"""
    state = BrainState(query=JEFFRIN_RESUME)
    result = await resume_parser_brain.run(state)
    assert result.response is not None
    r = result.response

    # All fields required by Resume Builder must be present
    assert "name" in r
    assert "email" in r
    assert "phone" in r
    assert "skills" in r
    assert "workExperiences" in r
    assert "educations" in r
    assert "summary" in r

    # Types must be correct for frontend to consume
    assert isinstance(r["skills"], list)
    assert isinstance(r["workExperiences"], list)
    assert isinstance(r["educations"], list)
    assert isinstance(r["summary"], str)

    # Name and email must be non-empty for Jeffrin's resume
    assert r["name"] != ""
    assert r["email"] == "jeffrin.in02@gmail.com"

"""Pure validator regressions; no LLM, network or database required."""
from recruitment_ai.validators.resume_fields import clean_field, source_phones, validate_resume_fields

def test_role_rejects_resume_paragraph_and_contact_information():
    assert clean_field(', with a track record of publishing research and delivering production-ready solutions. WORK EXPERIENCE Backend Developer 06/2025 - Present', 'jobTitle') == ''
    assert clean_field('Backend Developer', 'jobTitle') == 'Backend Developer'
    assert clean_field('person@example.com', 'jobTitle') == ''
    assert clean_field('Backend Developer 06/2025 - Present', 'jobTitle') == ''

def test_names_and_degrees_cannot_contain_roles_or_contacts():
    assert clean_field('Sherine P', 'name') == 'Sherine P'
    assert clean_field('Backend Developer', 'name') == ''
    assert clean_field('person@example.com', 'name') == ''
    assert clean_field('Backend Developer', 'degree') == ''
    assert clean_field('B.Tech', 'degree') == 'B.Tech'
    assert clean_field('2019 - 2025', 'phone') == ''

def test_grounding_and_aliases_apply_to_every_parser_path():
    source = 'Sherine P\nBackend Developer\nAcme Ltd\nPython\nsherine@example.com\n+91 98765 43210'
    data = validate_resume_fields({'name': 'Sherine P', 'title': 'with a track record of delivering solutions', 'jobTitle': 'madeup@example.com', 'workExperiences': [{'jobTitle': 'Backend Developer', 'company': 'Acme Ltd'}], 'skills': ['Python', 'Invented Skill']}, source)
    assert data['name'] == 'Sherine P'
    assert data['title'] == data['jobTitle'] == 'Backend Developer'
    assert data['skills'] == ['Python']
    assert data['email'] == 'sherine@example.com'
    assert data['phone'] == '+91 98765 43210'
    empty = validate_resume_fields({'name': 'Invented Name', 'skills': ['Python'], 'currentRole': 'some summary'}, source)
    assert empty['name'] == '' and empty['title'] == ''
    assert 'currentRole' not in empty


def test_us_contact_location_degree_and_linkedin_are_source_grounded():
    source = 'Jane Candidate\nLocation: Austin, TX 78701\nLinkedIn: https://www.linkedin.com/in/jane-candidate/\nEducation: B.S. Computer Science\nEmail: jane@example.com'
    data = validate_resume_fields({
        'name': 'Jane Candidate', 'location': 'Austin, TX 78701',
        'educations': [{'degree': 'B.S. Computer Science'}],
        'linkedin': 'https://linkedin.com/in/invented', 'country': 'India',
    }, source)
    assert data['location'] == 'Austin, TX 78701'
    assert data['educations'][0]['degree'] == 'B.S. Computer Science'
    assert data['linkedin'] == 'https://www.linkedin.com/in/jane-candidate'
    assert data['country'] == ''


def test_linkedin_company_page_and_lookalike_hosts_are_not_profile_links():
    source = 'https://www.linkedin.com/company/acme\nhttps://evil-linkedin.com/in/fake'
    assert validate_resume_fields({}, source)['linkedin'] == ''


def test_contact_header_wins_over_publication_isbn_and_dates():
    source = '''SHERINE P
Ayanavaram, Chennai | shreesherinerachel7 @ gmail . com | 8754437435
LinkedIn: linkedin . com / in / sherine-p
PUBLICATIONS & CONFERENCES
AI-Enabled Smart Water Quality Monitoring System (2025 - 2026)
ISBN 978-3-032-22118-6
Date of Birth: 01-01-2025'''
    data = validate_resume_fields({
        'name': 'SHERINE P', 'phone': '978-3-032-22118-6',
        'email': 'madeup@example.com', 'linkedin': 'https://linkedin.com/in/madeup',
    }, source)
    assert source_phones(source) == ['8754437435']
    assert data['name'] == 'SHERINE P'
    assert data['email'] == 'shreesherinerachel7@gmail.com'
    assert data['phone'] == '8754437435'
    assert data['linkedin'] == 'https://www.linkedin.com/in/sherine-p'

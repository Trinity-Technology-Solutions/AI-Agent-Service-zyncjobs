"""Pure validator regressions; no LLM, network or database required."""
from recruitment_ai.validators.resume_fields import clean_field, validate_resume_fields

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

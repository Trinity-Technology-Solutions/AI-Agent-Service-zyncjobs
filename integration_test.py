import requests
import json
import sys
import time

BASE = "http://localhost:8000"
USER_ID = "integration_test_user"
PASS = 0
FAIL = 0

def test(name, method, path, payload=None, expected_status=200):
    global PASS, FAIL
    url = f"{BASE}{path}"
    try:
        if method == "GET":
            r = requests.get(url, timeout=120)
        else:
            r = requests.post(url, json=payload, timeout=120)
        status = r.status_code
        if status == expected_status:
            try:
                data = r.json()
                PASS += 1
                print(f"  PASS | {name} | {status}")
                return data
            except:
                FAIL += 1
                print(f"  FAIL | {name} | {status} (not JSON)")
                return None
        else:
            FAIL += 1
            print(f"  FAIL | {name} | expected {expected_status}, got {status}")
            try:
                print(f"        body: {r.text[:200]}")
            except:
                pass
            return None
    except requests.exceptions.Timeout:
        FAIL += 1
        print(f"  FAIL | {name} | TIMEOUT (>120s)")
        return None
    except Exception as e:
        FAIL += 1
        print(f"  FAIL | {name} | {str(e)[:100]}")
        return None


print("=" * 60)
print("ZyncJobs AI Service — Integration Tests")
print("=" * 60)

# --- Health ---
print("\n[1] Health & Info")
test("Health", "GET", "/health")

# --- Candidate Flow ---
print("\n[2] Candidate Flow — Resume Parse")
SAMPLE_RESUME = """John Doe
john.doe@email.com
(555) 123-4567
linkedin.com/in/johndoe

Professional Summary
Experienced software engineer with 5+ years building scalable web applications.

Work Experience
Senior Software Engineer | Tech Corp
Jan 2020 - Present
- Led a team of 5 engineers to deliver a microservices platform
- Improved API response time by 40% through caching optimization
- Mentored 3 junior developers

Software Engineer | Startup Inc
Jun 2017 - Dec 2019
- Built RESTful APIs using Python and Django
- Reduced deployment time by 60% with CI/CD pipeline

Education
Master of Science in Computer Science
Stanford University, 2017

Skills
Python, Django, PostgreSQL, Docker, AWS, React, TypeScript, Redis, Git, CI/CD"""

parse = test("Parse Resume", "POST", "/api/v1/resume/parse", {
    "resume_text": SAMPLE_RESUME
})

if parse:
    for key in ["contact", "summary", "experience", "education", "skills"]:
        val = parse.get(key, "")
        if val:
            print(f"        {key}: {val[:60]}...")
        else:
            print(f"        {key}: (empty)")

print("\n[3] Candidate Flow — ATS Score")
ats = test("ATS Score", "POST", "/api/v1/resume/ats-score", {
    "resume_text": SAMPLE_RESUME,
    "job_description": "Senior Python Developer with Django, PostgreSQL, Docker, and AWS experience. Must have strong leadership skills and experience mentoring teams."
})

if ats:
    print(f"        score: {ats.get('score')}")
    print(f"        matching: {ats.get('matching_keywords', [])[:5]}")
    print(f"        missing: {ats.get('missing_keywords', [])[:5]}")

print("\n[4] Candidate Flow — Resume Improve")
improve = test("Improve Resume", "POST", "/api/v1/resume/improve", {
    "resume_text": SAMPLE_RESUME,
    "job_description": "Senior Python Developer with Django, PostgreSQL, Docker, and AWS experience."
})

if improve:
    print(f"        ats_score from tool: {improve.get('ats_score')}")
    print(f"        skills_suggested: {improve.get('skills_suggested', [])[:5]}")
    print(f"        grammar_issues count: {len(improve.get('grammar_issues', []))}")
    improved = improve.get("improved_resume", "")
    print(f"        improved resume length: {len(improved)} chars")
    # Check no RAG context leaking
    if "Relevant Knowledge" in improved or "Relevant Knowledge" in improved:
        print("        WARNING: RAG context leaking into output!")
    else:
        print("        OK: No RAG context leakage")

print("\n[5] Candidate Flow — Job Match")
match = test("Job Match", "POST", "/api/v1/job/match", {
    "resume_text": SAMPLE_RESUME,
    "job_description": "Senior Python Developer needed with Django, PostgreSQL, Docker, AWS, and React experience."
})

if match:
    print(f"        match_score: {match.get('match_score')}")
    print(f"        matching_skills: {match.get('matching_skills', [])[:5]}")
    print(f"        missing_skills: {match.get('missing_skills', [])[:5]}")

print("\n[6] Candidate Flow — Career Advice")
career = test("Career Advice", "POST", "/api/v1/career/advice", {
    "current_role": "Junior Python Developer",
    "target_role": "Senior Software Architect",
    "skills": ["Python", "Django", "PostgreSQL", "Docker"]
})

if career:
    advice = career.get("advice", "")
    print(f"        advice length: {len(advice)} chars")
    if "Relevant Knowledge" in advice or "knowledge" in advice.lower()[:100]:
        print("        WARNING: RAG context may be leaking")
    else:
        print("        OK: No obvious RAG leakage")

print("\n[7] Candidate Flow — Interview Questions")
interview = test("Interview Questions", "POST", "/api/v1/interview/questions", {
    "job_title": "Senior Python Developer",
    "skills": ["Python", "Django", "PostgreSQL", "System Design"],
    "experience_level": "senior"
})

if interview:
    questions = interview.get("questions", "")
    print(f"        questions length: {len(questions)} chars")

print("\n[8] Candidate Flow — AI Chat")
chat1 = test("Chat (resume question)", "POST", "/api/v1/chat", {
    "message": "What skills should I add to my resume for a senior Python role?",
    "user_id": USER_ID
})

if chat1:
    print(f"        reply length: {len(chat1.get('reply', ''))} chars")
    print(f"        intent: {chat1.get('intent')}")

chat2 = test("Chat (follow-up question)", "POST", "/api/v1/chat", {
    "message": "Can you suggest some learning resources for those skills?",
    "user_id": USER_ID
})

if chat2:
    print(f"        reply length: {len(chat2.get('reply', ''))} chars")
    print(f"        intent: {chat2.get('intent')}")

# --- Recruiter Flow ---
print("\n[9] Recruiter Flow — Generate JD")
jd = test("Generate JD", "POST", "/api/v1/recruiter/generate-jd", {
    "title": "Senior Python Developer",
    "experience_level": "Senior",
    "skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS", "React"]
})

if jd:
    desc = jd.get("job_description", "")
    print(f"        JD length: {len(desc)} chars")
    if "Relevant Knowledge" in desc:
        print("        WARNING: RAG context leaking into JD output!")
    else:
        print("        OK: No RAG leakage")

# --- Knowledge Base ---
print("\n[10] Knowledge Base")
test("Knowledge Stats", "GET", "/api/v1/knowledge/stats")

kb_query = test("Knowledge Query", "POST", "/api/v1/knowledge/query", {
    "query": "best practices for resume writing"
})

if kb_query:
    results = kb_query.get("results", kb_query.get("documents", []))
    print(f"        results count: {len(results) if isinstance(results, list) else 'unknown'}")

# --- Summary ---
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 60)

if FAIL == 0:
    print("All integration tests passed!")
    sys.exit(0)
else:
    print(f"{FAIL} test(s) failed!")
    sys.exit(1)

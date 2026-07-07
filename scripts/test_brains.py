"""Test all brain intents end-to-end."""
import httpx
import json

base = "http://localhost:8001"

# Get JWT token
r = httpx.post(f"{base}/auth/token", json={"user_id": "test_user", "role": "candidate"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

tests = [
    ("CHAT", {"query": "hi hello what can you do"}),
    ("JOB_MATCH", {"query": "find me a job as python developer"}),
    ("ATS_SCORE", {"query": "check my ats score"}),
    ("CAREER_ADVICE", {"query": "career advice for software engineer"}),
    ("JD_GENERATOR", {"query": "generate a job description for a react developer"}),
    ("RESUME_PARSER", {"query": "parse this resume text"}),
    ("RECRUITER", {"query": "find candidates for react developer"}),
    ("INTERVIEW_PREP", {"query": "interview preparation tips"}),
    ("SKILL_ASSESSMENT", {"query": "take a skill assessment test"}),
    ("RESUME_BUILDER", {"query": "build a resume for me"}),
    ("JOB_PARSER", {"query": "parse this job description"}),
    ("RECRUITER_SHORTLIST", {"query": "shortlist top candidates"}),
]

for name, payload in tests:
    try:
        r = httpx.post(f"{base}/ai/execute", json=payload, headers=headers, timeout=180)
        data = r.json()
        status = "PASS" if data.get("success") else "FAIL"
        fb = " [FALLBACK]" if data.get("result", {}).get("fallback") else ""
        err = data.get("error", "")
        intent = data.get("intent", "")
        result = data.get("result", {})
        res_str = json.dumps(result)[:120] if result else "None"
        print(f"{status}: {name} -> intent={intent} error={err}{fb}")
        if result:
            print(f"  result: {res_str}")
    except Exception as e:
        print(f"ERROR: {name} - {e}")

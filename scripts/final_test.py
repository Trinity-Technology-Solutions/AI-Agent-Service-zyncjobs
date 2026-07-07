"""Final end-to-end test of all AI brains."""
import httpx
import json

base = "http://localhost:8001"

r = httpx.post(f"{base}/auth/token", json={"user_id": "test_user", "role": "candidate"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

tests = [
    ("1.CHAT", {"query": "hi hello what can you do"}),
    ("2.JOB_MATCH", {"query": "match jobs", "context": {"candidate_profile": "Python dev 5yr Django React", "job_requirements": "Senior Python Django AWS React 3yr+"}}),
    ("3.ATS_SCORE", {"query": "check my ats score"}),
    ("4.CAREER_ADVICE", {"query": "career advice for software engineer"}),
    ("5.JD_GENERATOR", {"query": "generate a job description for a react developer"}),
    ("6.RESUME_PARSER", {"query": "parse this resume text"}),
    ("7.RECRUITER", {"query": "find candidates for react developer"}),
    ("8.INTERVIEW_PREP", {"query": "interview preparation tips"}),
    ("9.SKILL_ASSESS", {"query": "take a skill assessment test"}),
    ("10.RESUME_BUILDER", {"query": "build a resume for me"}),
    ("11.JOB_PARSER", {"query": "parse this job description"}),
    ("12.RECRUITER_SHORT", {"query": "shortlist top candidates"}),
]

results = []
for name, payload in tests:
    try:
        r = httpx.post(f"{base}/ai/execute", json=payload, headers=headers, timeout=180)
        data = r.json()
        success = data.get("success", False)
        intent = data.get("intent", "?")
        error = data.get("error")
        result = data.get("result")
        fb = result.get("fallback", False) if isinstance(result, dict) else False
        status = "[OK]" if success else "[FAIL]"
        fb_str = " FALLBACK" if fb else ""
        err_str = f" error={error}" if error else ""
        keys = list(result.keys())[:3] if isinstance(result, dict) else []
        results.append(f"{status} {name} -> {intent}{fb_str}{err_str} keys={keys}")
    except Exception as e:
        results.append(f"💥 {name} → CRASH: {e}")

print("\n".join(results))
print(f"\nTotal: {len(tests)} tests")

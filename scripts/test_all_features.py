"""Test all AI features via API (simulating frontend calls)."""
import asyncio
import httpx
import json

AI_BASE = "http://localhost:8001"

async def get_token():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{AI_BASE}/auth/token", 
            json={"user_id": "test_user", "role": "candidate"}, timeout=10)
        return r.json()["access_token"]

async def execute(token, query, context=None, user_role="candidate"):
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"query": query, "user_role": user_role}
        if context:
            payload["context"] = context
        r = await client.post(f"{AI_BASE}/ai/execute", json=payload, headers=headers, timeout=180)
        return r.json()

async def test_all():
    token = await get_token()
    print(f"Got token: {token[:20]}...\n")
    
    tests = [
        # (name, query, context, expected_intent)
        ("1. CHAT", "hi what is zyncjobs", None, "CHAT"),
        ("2. JOB_PARSER", "parse job: Senior Python Developer, 5yrs exp, Django, AWS", None, "JOB_PARSER"),
        ("3. JD_GENERATOR", "generate job description for React developer", None, "JD_GENERATOR"),
        ("4. RESUME_PARSER", "parse resume: John Doe, Python dev, 3yrs exp", None, "RESUME_PARSER"),
        ("5. ATS_SCORE", "check my ats score", {"resume": "Python developer", "job_description": "Python Django"}, "ATS_SCORE"),
        ("6. JOB_MATCH", "match jobs", {"candidate_profile": "Python Django 5yrs", "job_requirements": "Senior Python AWS 3yrs"}, "JOB_MATCH"),
        ("7. CAREER_ADVICE", "career advice for software engineer", None, "CAREER_ADVICE"),
        ("8. SKILL_ASSESSMENT", "take skill assessment for Python", None, "SKILL_ASSESSMENT"),
        ("9. INTERVIEW_PREP", "interview preparation for senior developer", None, "INTERVIEW_PREP"),
        ("10. RESUME_BUILDER", "build resume for software engineer", None, "RESUME_BUILDER"),
        ("11. RECRUITER", "find candidates for React developer", None, "RECRUITER"),
        ("12. RECRUITER_SHORTLIST", "shortlist top candidates", {"job_requirements": "Python Django 5yrs", "candidates": [{"name": "A", "skills": ["Python", "Django"]}]}, "RECRUITER_SHORTLIST"),
    ]
    
    results = []
    for name, query, context, expected_intent in tests:
        try:
            data = await execute(token, query, context)
            success = data.get("success", False)
            intent = data.get("intent", "?")
            error = data.get("error")
            result = data.get("result", {})
            fallback = result.get("fallback", False) if isinstance(result, dict) else False
            fb_str = " ⚠️ FALLBACK" if fallback else ""
            err_str = f" ERROR: {error}" if error else ""
            intent_match = " ✓" if intent == expected_intent else f" ✗ (got {intent})"
            keys = list(result.keys())[:4] if isinstance(result, dict) else []
            status = "[OK]" if success else "[FAIL]"
            results.append(f"{status} {name} -> {intent}{fb_str}{err_str}{intent_match} keys={keys}")
        except Exception as e:
            results.append(f"[CRASH] {name} -> {e}")
    
    print("\n".join(results))
    print(f"\nSummary: {sum(1 for r in results if r.startswith('[OK]'))}/{len(results)} passed")

asyncio.run(test_all())
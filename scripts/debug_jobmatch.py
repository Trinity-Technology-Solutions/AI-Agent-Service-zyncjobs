"""Test JOB_MATCH with proper context and debug all brains."""
import httpx
import json

base = "http://localhost:8001"

r = httpx.post(f"{base}/auth/token", json={"user_id": "test_user", "role": "candidate"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Test JOB_MATCH with proper context
payload = {
    "query": "match jobs",
    "context": {
        "candidate_profile": "Python developer with 5 years experience in Django, React, PostgreSQL",
        "job_requirements": "Senior Python developer, 3+ years, Django, AWS, React, SQL"
    }
}
r = httpx.post(f"{base}/ai/execute", json=payload, headers=headers, timeout=180)
data = r.json()
print(f"JOB_MATCH (with context): success={data.get('success')} intent={data.get('intent')} error={data.get('error')}")
print(f"  result: {json.dumps(data.get('result'))[:200]}")

# Test without context to see error path
payload2 = {"query": "find me a python job"}
r2 = httpx.post(f"{base}/ai/execute", json=payload2, headers=headers, timeout=180)
data2 = r2.json()
print(f"\nJOB_MATCH (no context): success={data2.get('success')} intent={data2.get('intent')} error={data2.get('error')}")
print(f"  result: {json.dumps(data2.get('result'))}")

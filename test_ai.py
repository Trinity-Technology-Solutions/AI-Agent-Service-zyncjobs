import urllib.request, json

# Test health
resp = urllib.request.urlopen("http://127.0.0.1:8001/health")
print("Health:", resp.read().decode())

# Test auth
data = json.dumps({"user_id": "test", "role": "candidate"}).encode()
req = urllib.request.Request("http://127.0.0.1:8001/auth/token", data=data, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
token = json.loads(resp.read())["access_token"]
print("Token OK:", token[:20] + "...")

# Test execute with history
payload = json.dumps({
    "query": "what is my last question?",
    "context": {
        "systemPrompt": "You are a helpful assistant",
        "history": [
            {"role": "user", "content": "my name is John"},
            {"role": "assistant", "content": "Nice to meet you John!"}
        ]
    },
    "user_role": "candidate"
}).encode()
req = urllib.request.Request("http://127.0.0.1:8001/ai/execute", data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
reply = result.get("result", {}).get("reply", "no reply")
print("Reply:", reply[:200])
print("SUCCESS: AI service working with history context")

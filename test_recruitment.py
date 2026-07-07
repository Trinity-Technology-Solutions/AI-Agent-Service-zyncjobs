"""Test the recruitment AI platform."""
import httpx, asyncio, json

async def test():
    async with httpx.AsyncClient() as c:
        # 1. Token
        r = await c.post('http://localhost:8001/auth/token', json={'user_id':'test','role':'candidate'})
        token = r.json()['access_token']
        h = {'Authorization': f'Bearer {token}'}
        print(f'1. Token: [{r.status_code}] OK')

        # 2. Root
        r = await c.get('http://localhost:8001/')
        data = r.json()
        print(f'2. Root: [{r.status_code}] {data.get("name")}')

        # 3. Health
        r = await c.get('http://localhost:8001/health')
        data = r.json()
        print(f'3. Health: [{r.status_code}] status={data.get("status")}')

        # 4. Knowledge stats
        r = await c.get('http://localhost:8001/knowledge/stats')
        data = r.json()
        print(f'4. Knowledge: [{r.status_code}] {data.get("total_chunks")} chunks')

        # 5. Version
        r = await c.get('http://localhost:8001/version')
        data = r.json()
        print(f'5. Version: [{r.status_code}] brains={data.get("brains")}')

        # 6. Chat
        r = await c.post('http://localhost:8001/ai/execute', json={'query':'What is ZyncJobs?'}, headers=h)
        data = r.json()
        if data.get("success"):
            result = data.get("result", {})
            reply_len = len(result.get("reply", "") or "")
            print(f'6. Chat: [{r.status_code}] intent={data.get("intent")} reply={reply_len} chars')
        else:
            print(f'6. Chat: [{r.status_code}] error={data.get("error")}')

        # 7. ATS Score
        r = await c.post('http://localhost:8001/ai/execute', json={'query':'Check my ATS score for Python developer role'}, headers=h)
        data = r.json()
        print(f'7. ATS: [{r.status_code}] intent={data.get("intent")}')

        # 8. Recruiter
        r = await c.post('http://localhost:8001/ai/execute', json={'query':'Find Python developers for a senior role'}, headers=h)
        data = r.json()
        print(f'8. Recruiter: [{r.status_code}] intent={data.get("intent")}')

        # 9. Metrics
        r = await c.get('http://localhost:8001/metrics')
        print(f'9. Metrics: [{r.status_code}] len={len(r.text)}')

asyncio.run(test())

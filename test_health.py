"""Quick test of health endpoint."""
import httpx

r = httpx.get('http://localhost:8001/health')
print(f'Status: {r.status_code}')
print(f'Headers: {dict(r.headers)}')
print(f'Text: {r.text[:500]}')

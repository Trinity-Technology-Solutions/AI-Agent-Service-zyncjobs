"""HTTP client for calling the ZyncJobs Node backend APIs.
Replaces direct database access with proper service-to-service calls.

Architecture:
  AI Service (FastAPI, port 8001)
       │  HTTP
       ▼
  Node Backend (Express, port 5000)
       │
       ▼
  PostgreSQL / S3 / Redis (owned by backend)
"""
import logging
import httpx
from typing import Optional
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)


class BackendClient:
    """Thin HTTP client that calls the Node backend APIs.
    Every method returns dicts from the backend JSON response body.
    Lazy initialization — client created on first call.
    """

    def __init__(self, base_url: str = ""):
        self.base_url = base_url or settings.BACKEND_API_URL
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=3.0,
                    read=settings.BACKEND_TIMEOUT,
                    write=settings.BACKEND_TIMEOUT,
                    pool=3.0,
                ),
                headers={"User-Agent": "zyncjobs-ai-service/1.0"},
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Auth ──────────────────────────────────────────────────────────

    async def get_user(self, user_id: str) -> Optional[dict]:
        """GET /api/users/:id — returns user profile."""
        try:
            resp = await self.client.get(f"/api/users/{user_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Backend get_user(%s) failed: %s", user_id, e)
            return None

    # ── Resume ────────────────────────────────────────────────────────

    async def get_resume(self, user_id: str) -> Optional[dict]:
        """GET /api/resume?userId=:id — returns parsed resume data."""
        try:
            resp = await self.client.get("/api/resume", params={"userId": user_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Backend get_resume(%s) failed: %s", user_id, e)
            return None

    async def parse_resume(self, file_url: str) -> Optional[dict]:
        """POST /api/resume-parser/parse — parse a resume file."""
        try:
            resp = await self.client.post("/api/resume-parser/parse", json={"url": file_url})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Backend parse_resume failed: %s", e)
            return None

    async def get_resume_versions(self, user_id: str) -> list[dict]:
        """GET /api/resume-versions?userId=:id"""
        try:
            resp = await self.client.get("/api/resume-versions", params={"userId": user_id})
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Backend get_resume_versions(%s) failed: %s", user_id, e)
            return []

    # ── Company ───────────────────────────────────────────────────────

    async def get_company(self, name: str) -> Optional[dict]:
        """GET /api/companies?name=:name"""
        try:
            resp = await self.client.get("/api/companies", params={"name": name})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Backend get_company(%s) failed: %s", name, e)
            return None

    # ── Jobs ──────────────────────────────────────────────────────────

    async def get_job(self, job_id: str) -> Optional[dict]:
        """GET /api/jobs/:id"""
        try:
            resp = await self.client.get(f"/api/jobs/{job_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Backend get_job(%s) failed: %s", job_id, e)
            return None

    async def search_jobs(self, query: str, limit: int = 10) -> list[dict]:
        """GET /api/jobs?search=:query&limit=:limit"""
        try:
            resp = await self.client.get("/api/jobs", params={"search": query, "limit": limit})
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Backend search_jobs(%s) failed: %s", query, e)
            return []

    # ── Assessments ───────────────────────────────────────────────────

    async def get_assessment(self, user_id: str) -> Optional[dict]:
        """GET /api/skill-assessments?userId=:id"""
        try:
            resp = await self.client.get("/api/skill-assessments", params={"userId": user_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Backend get_assessment(%s) failed: %s", user_id, e)
            return None

    # ── Conversations (memory) ────────────────────────────────────────

    async def get_conversation_history(self, session_id: str) -> list[dict]:
        """GET /api/messages?conversationId=:id"""
        try:
            resp = await self.client.get("/api/messages", params={"conversationId": session_id})
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Backend get_conversation_history(%s) failed: %s", session_id, e)
            return []

    # ── Candidates (for recruiter) ────────────────────────────────────

    async def search_candidates(self, criteria: str, limit: int = 20) -> list[dict]:
        """GET /api/candidates?q=:criteria&limit=:limit"""
        try:
            resp = await self.client.get("/api/candidates", params={"q": criteria, "limit": limit})
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Backend search_candidates(%s) failed: %s", criteria, e)
            return []

    # ── Health ────────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """GET /api/health — verify backend is reachable."""
        try:
            resp = await self.client.get("/api/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


backend_client = BackendClient()

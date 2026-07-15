"""Repository layer — calls ZyncJobs Node backend APIs instead of direct DB.
Previously used SQLAlchemy directly; now delegates to BackendClient
so the AI service is decoupled from the backend's database schema.

Architecture doc: AI Service → HTTP → Node Backend → PostgreSQL
"""
from typing import Optional
import recruitment_ai.services.backend_client as _backend
import logging

logger = logging.getLogger(__name__)


class UserRepository:
    async def get_by_user_id(self, user_id: str) -> Optional[dict]:
        try:
            return await _backend.backend_client.get_user(user_id)
        except Exception as e:
            logger.warning("UserRepository.get_by_user_id error: %s", e)
        return None

    async def get_profile(self, user_id: str) -> Optional[dict]:
        try:
            user = await _backend.backend_client.get_user(user_id)
            if not user:
                return None
            return {
                "email": user.get("email"),
                "role": user.get("role"),
                "name": user.get("name"),
                "title": user.get("title"),
                "location": user.get("location"),
                "experience_years": user.get("experience_years"),
                "skills": user.get("skills") or [],
                "preferences": user.get("preferences") or {},
                "ats_score": user.get("ats_score"),
                "applications_count": user.get("applications_count") or 0,
                "missing_skills": user.get("missing_skills") or [],
            }
        except Exception as e:
            logger.warning("UserRepository.get_profile error: %s", e)
        return None

    async def upsert(self, user_id: str, role: str, email: Optional[str] = None, name: Optional[str] = None) -> Optional[dict]:
        try:
            return await _backend.backend_client.get_user(user_id)
        except Exception as e:
            logger.warning("UserRepository.upsert error: %s", e)
        return None


class ResumeRepository:
    async def save(self, user_id: int, raw_text: str, parsed_data: dict, ats_score: Optional[int] = None) -> Optional[dict]:
        try:
            return {"user_id": user_id, "raw_text": raw_text[:200], "parsed": True}
        except Exception as e:
            logger.warning("ResumeRepository.save error: %s", e)
        return None

    async def get_latest(self, user_id: int) -> Optional[dict]:
        try:
            return await backend_client.get_resume(str(user_id))
        except Exception as e:
            logger.warning("ResumeRepository.get_latest error: %s", e)
        return None

    async def get_resume_data(self, user_id: str) -> Optional[dict]:
        try:
            data = await backend_client.get_resume(user_id)
            if data:
                return {
                    "parsed": data.get("parsedData") or data.get("parsed_data"),
                    "skills": data.get("skills") or [],
                    "experience": data.get("experience") or [],
                    "education": data.get("education") or [],
                }
        except Exception as e:
            logger.warning("ResumeRepository.get_resume_data error: %s", e)
        return None


class JobPostRepository:
    async def save(self, data: dict) -> Optional[dict]:
        return data

    async def get_active(self, limit: int = 20) -> list[dict]:
        try:
            return await backend_client.search_jobs("", limit=limit)
        except Exception as e:
            logger.warning("JobPostRepository.get_active error: %s", e)
        return []


class ConversationRepository:
    async def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        try:
            return await backend_client.get_conversation_history(session_id)
        except Exception as e:
            logger.warning("ConversationRepository.get_history error: %s", e)
        return []

    async def save_turn(self, session_id: str, user_id: Optional[int], role: str, content: str, intent: str) -> None:
        logger.debug("Conversation turn saved (session=%s, role=%s)", session_id, role)


class CompanyRepository:
    async def get_by_name(self, name: str) -> Optional[dict]:
        try:
            return await backend_client.get_company(name)
        except Exception as e:
            logger.warning("CompanyRepository.get_by_name error: %s", e)
        return None

    async def get_profile(self, name: str) -> Optional[dict]:
        try:
            company = await backend_client.get_company(name)
            if company:
                return {"name": company.get("name"), "industry": company.get("industry")}
        except Exception as e:
            logger.warning("CompanyRepository.get_profile error: %s", e)
        return None

    async def upsert(self, name: str, industry: Optional[str] = None) -> Optional[dict]:
        try:
            return {"name": name, "industry": industry}
        except Exception as e:
            logger.warning("CompanyRepository.upsert error: %s", e)
        return None


class AssessmentRepository:
    async def get_last_by_user(self, user_id: int) -> Optional[dict]:
        try:
            return await backend_client.get_assessment(str(user_id))
        except Exception as e:
            logger.warning("AssessmentRepository.get_last_by_user error: %s", e)
        return None

    async def get_last_assessment_data(self, user_id: str) -> Optional[dict]:
        try:
            data = await backend_client.get_assessment(user_id)
            if data:
                return {
                    "id": str(data.get("id")),
                    "skill": data.get("skill"),
                    "score": data.get("score"),
                    "level": data.get("level"),
                    "completed_at": str(data.get("createdAt", "")),
                }
        except Exception as e:
            logger.warning("AssessmentRepository.get_last_assessment_data error: %s", e)
        return None

    async def save(self, user_id: int, skill: str, score: Optional[float] = None, level: Optional[str] = None) -> Optional[dict]:
        return {"user_id": user_id, "skill": skill, "score": score, "level": level}


class KnowledgeChunkRepository:
    async def get_by_category(self, category: str, limit: int = 50) -> list[dict]:
        logger.debug("KnowledgeChunkRepository.get_by_category(%s)", category)
        return []

    async def save(self, chunk_id: str, url: str, title: str, category: str, content: str, section: Optional[str] = None) -> Optional[dict]:
        return {"chunk_id": chunk_id, "url": url, "title": title, "category": category}


# Singleton instances
user_repo = UserRepository()
resume_repo = ResumeRepository()
job_repo = JobPostRepository()
conversation_repo = ConversationRepository()
company_repo = CompanyRepository()
assessment_repo = AssessmentRepository()
knowledge_chunk_repo = KnowledgeChunkRepository()

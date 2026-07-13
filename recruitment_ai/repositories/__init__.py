"""Repository layer — abstracts all DB access from brains and services.
Architecture doc: repositories/ pattern for PostgreSQL queries.
"""
from typing import Optional
from sqlalchemy import select, desc
from recruitment_ai.database.connection import get_db
from recruitment_ai.database.models import User, Resume, JobPost, Conversation, KnowledgeChunk
import logging

logger = logging.getLogger(__name__)


class UserRepository:
    async def get_by_user_id(self, user_id: str) -> Optional[User]:
        try:
            async for db in get_db():
                result = await db.execute(select(User).where(User.user_id == user_id))
                return result.scalar_one_or_none()
        except Exception as e:
            logger.warning("UserRepository.get_by_user_id error: %s", e)
        return None

    async def upsert(self, user_id: str, role: str, email: Optional[str] = None) -> Optional[User]:
        try:
            async for db in get_db():
                result = await db.execute(select(User).where(User.user_id == user_id))
                user = result.scalar_one_or_none()
                if not user:
                    user = User(user_id=user_id, role=role, email=email)
                    db.add(user)
                return user
        except Exception as e:
            logger.warning("UserRepository.upsert error: %s", e)
        return None


class ResumeRepository:
    async def save(self, user_id: int, raw_text: str, parsed_data: dict, ats_score: Optional[int] = None) -> Optional[Resume]:
        try:
            async for db in get_db():
                resume = Resume(user_id=user_id, raw_text=raw_text, parsed_data=parsed_data, ats_score=ats_score)
                db.add(resume)
                return resume
        except Exception as e:
            logger.warning("ResumeRepository.save error: %s", e)
        return None

    async def get_latest(self, user_id: int) -> Optional[Resume]:
        try:
            async for db in get_db():
                result = await db.execute(
                    select(Resume).where(Resume.user_id == user_id).order_by(desc(Resume.created_at)).limit(1)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.warning("ResumeRepository.get_latest error: %s", e)
        return None


class JobPostRepository:
    async def save(self, data: dict) -> Optional[JobPost]:
        try:
            async for db in get_db():
                job = JobPost(**{k: v for k, v in data.items() if hasattr(JobPost, k)})
                db.add(job)
                return job
        except Exception as e:
            logger.warning("JobPostRepository.save error: %s", e)
        return None

    async def get_active(self, limit: int = 20) -> list[JobPost]:
        try:
            async for db in get_db():
                result = await db.execute(
                    select(JobPost).where(JobPost.is_active == True).order_by(desc(JobPost.created_at)).limit(limit)
                )
                return list(result.scalars().all())
        except Exception as e:
            logger.warning("JobPostRepository.get_active error: %s", e)
        return []


class ConversationRepository:
    async def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        try:
            async for db in get_db():
                result = await db.execute(
                    select(Conversation)
                    .where(Conversation.session_id == session_id)
                    .order_by(desc(Conversation.created_at))
                    .limit(limit)
                )
                rows = list(reversed(result.scalars().all()))
                return [{"role": r.role, "content": r.content} for r in rows]
        except Exception as e:
            logger.warning("ConversationRepository.get_history error: %s", e)
        return []

    async def save_turn(self, session_id: str, user_id: Optional[int], role: str, content: str, intent: str) -> None:
        try:
            async for db in get_db():
                db.add(Conversation(
                    session_id=session_id,
                    user_id=user_id,
                    role=role,
                    content=content,
                    metadata_json={"intent": intent},
                ))
        except Exception as e:
            logger.warning("ConversationRepository.save_turn error: %s", e)


# Singleton instances
user_repo = UserRepository()
resume_repo = ResumeRepository()
job_repo = JobPostRepository()
conversation_repo = ConversationRepository()

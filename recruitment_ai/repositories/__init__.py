"""Data access repositories for database operations."""
from typing import Optional, List
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from recruitment_ai.database.models import User, JobPost, Resume, Conversation, KnowledgeChunk
from recruitment_ai.database.connection import async_session_factory, init_db
from datetime import datetime


class UserRepository:
    async def get_by_user_id(self, user_id: str) -> Optional[User]:
        async with async_session_factory() as session:
            result = await session.execute(select(User).where(User.user_id == user_id))
            return result.scalar_one_or_none()

    async def create(self, user_id: str, email: str = None, role: str = "candidate", name: str = None) -> User:
        async with async_session_factory() as session:
            user = User(user_id=user_id, email=email, role=role, name=name)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def update_profile(self, user_id: str, profile_data: dict) -> Optional[User]:
        async with async_session_factory() as session:
            result = await session.execute(select(User).where(User.user_id == user_id))
            user = result.scalar_one_or_none()
            if user:
                user.profile_data = profile_data
                user.updated_at = datetime.utcnow()
                await session.commit()
                await session.refresh(user)
            return user


class JobPostRepository:
    async def search(self, skills: list[str] = None, location: str = None, job_type: str = None, limit: int = 20) -> List[JobPost]:
        async with async_session_factory() as session:
            query = select(JobPost).where(JobPost.is_active == True)
            if location:
                query = query.where(JobPost.location.ilike(f"%{location}%"))
            if job_type:
                query = query.where(JobPost.job_type == job_type)
            query = query.limit(limit)
            result = await session.execute(query.order_by(JobPost.created_at.desc()))
            return list(result.scalars().all())

    async def create(self, data: dict) -> JobPost:
        async with async_session_factory() as session:
            job = JobPost(**data)
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job


class ResumeRepository:
    async def get_by_user(self, user_id: int) -> List[Resume]:
        async with async_session_factory() as session:
            result = await session.execute(select(Resume).where(Resume.user_id == user_id))
            return list(result.scalars().all())

    async def save(self, user_id: int, raw_text: str, parsed_data: dict = None) -> Resume:
        async with async_session_factory() as session:
            resume = Resume(user_id=user_id, raw_text=raw_text, parsed_data=parsed_data)
            session.add(resume)
            await session.commit()
            await session.refresh(resume)
            return resume


class ConversationRepository:
    async def get_by_session(self, session_id: str, limit: int = 20) -> List[Conversation]:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.asc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def add_message(self, session_id: str, role: str, content: str, user_id: int = None) -> Conversation:
        async with async_session_factory() as session:
            msg = Conversation(session_id=session_id, role=role, content=content, user_id=user_id)
            session.add(msg)
            await session.commit()
            await session.refresh(msg)
            return msg

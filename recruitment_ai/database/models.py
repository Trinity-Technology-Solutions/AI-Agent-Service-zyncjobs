"""SQLAlchemy ORM models for recruitment platform."""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from recruitment_ai.database.base import Base


class UserRole(str, enum.Enum):
    CANDIDATE = "candidate"
    EMPLOYER = "employer"
    ADMIN = "admin"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True)
    role = Column(String(20), default="candidate")
    name = Column(String(255), nullable=True)
    profile_data = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    conversations = relationship("Conversation", back_populates="user")
    resumes = relationship("Resume", back_populates="user")


class JobPost(Base):
    __tablename__ = "job_posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(100), unique=True, nullable=True)
    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    job_type = Column(String(50), default="full-time")
    experience_level = Column(String(50), default="mid")
    salary_min = Column(Float, nullable=True)
    salary_max = Column(Float, nullable=True)
    currency = Column(String(10), default="USD")
    description = Column(Text, nullable=True)
    skills_required = Column(JSON, nullable=True)
    skills_preferred = Column(JSON, nullable=True)
    responsibilities = Column(JSON, nullable=True)
    requirements = Column(JSON, nullable=True)
    benefits = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employer_id = Column(Integer, ForeignKey("users.id"), nullable=True)


class Resume(Base):
    __tablename__ = "resumes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), default="My Resume")
    raw_text = Column(Text, nullable=True)
    parsed_data = Column(JSON, nullable=True)
    ats_score = Column(Integer, nullable=True)
    file_path = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="resumes")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="conversations")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(String(100), unique=True, nullable=True)
    url = Column(String(500), nullable=True)
    title = Column(String(255), nullable=True)
    category = Column(String(100), nullable=True)
    section = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    embedding_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

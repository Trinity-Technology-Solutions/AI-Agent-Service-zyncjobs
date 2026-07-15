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
    title = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    experience_years = Column(Float, nullable=True)
    skills = Column(JSON, nullable=True)
    preferences = Column(JSON, nullable=True)
    ats_score = Column(Integer, nullable=True)
    applications_count = Column(Integer, default=0)
    missing_skills = Column(JSON, nullable=True)
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
    skills = Column(JSON, nullable=True)
    experience = Column(JSON, nullable=True)
    education = Column(JSON, nullable=True)
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


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    industry = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    logo_url = Column(String(500), nullable=True)
    website = Column(String(500), nullable=True)
    size = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    job_posts = relationship("JobPost", backref="company_ref", foreign_keys=[JobPost.company])


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    skill = Column(String(100), nullable=False)
    score = Column(Float, nullable=True)
    level = Column(String(50), nullable=True)
    questions = Column(JSON, nullable=True)
    answers = Column(JSON, nullable=True)
    completed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", backref="assessments")


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

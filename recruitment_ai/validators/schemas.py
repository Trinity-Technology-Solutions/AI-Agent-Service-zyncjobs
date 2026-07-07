"""Pydantic validators for brain inputs/outputs."""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


class ExecuteRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = None
    user_role: Literal["candidate", "employer", "admin"] = "candidate"
    context: Optional[dict] = None
    file_content: Optional[str] = None
    file_type: Optional[str] = None


class ExecuteResponse(BaseModel):
    success: bool
    intent: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    metadata: dict = {}


class JobDescription(BaseModel):
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    job_type: Literal["full-time", "part-time", "contract", "internship"] = "full-time"
    experience_level: Literal["entry", "mid", "senior", "lead", "executive"] = "mid"
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    currency: str = "USD"
    skills_required: List[str] = []
    skills_preferred: List[str] = []
    responsibilities: List[str] = []
    requirements: List[str] = []
    benefits: List[str] = []
    description: str


class ResumeData(BaseModel):
    personal_info: dict = {}
    summary: Optional[str] = None
    experience: List[dict] = []
    education: List[dict] = []
    skills: dict = {}
    projects: List[dict] = []
    certifications: List[dict] = []


class ATSScoreResult(BaseModel):
    ats_score: int = Field(ge=0, le=100)
    keyword_match: dict
    formatting_score: int = Field(ge=0, le=100)
    section_completeness: int = Field(ge=0, le=100)
    experience_relevance: int = Field(ge=0, le=100)
    suggestions: List[str] = []
    passes_ats: bool


class JobMatchResult(BaseModel):
    match_score: int = Field(ge=0, le=100)
    skill_match: dict
    experience_match: dict
    location_match: str
    salary_match: str
    strengths: List[str] = []
    gaps: List[str] = []
    recommendation: Literal["strong_match", "good_match", "potential_match", "poor_match"]


class CareerAdviceResult(BaseModel):
    career_path: List[dict] = []
    skill_gaps: List[str] = []
    recommended_courses: List[dict] = []
    certifications: List[str] = []
    timeline_months: int
    advice: str


class SkillAssessmentResult(BaseModel):
    questions: List[dict] = []


class InterviewPrepResult(BaseModel):
    questions: List[dict] = []
    topics_to_review: List[str] = []
    tips: List[str] = []


class ResumeBuilderResult(BaseModel):
    summary: str
    experience_bullets: List[dict] = []
    skills_formatted: dict = {}
    ats_keywords: List[str] = []


class ChatbotResult(BaseModel):
    reply: str
    sources: List[dict] = []
    intent: str
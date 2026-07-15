from pydantic import BaseModel
from typing import Optional


class ResumeParseRequest(BaseModel):
    file_content: str
    file_type: str = "pdf"
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ResumeParseResponse(BaseModel):
    success: bool
    parsed: Optional[dict] = None
    error: Optional[str] = None


class ResumeBuildRequest(BaseModel):
    sections: dict
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ResumeEditRequest(BaseModel):
    section: str
    instruction: str
    current_content: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class JobMatchRequest(BaseModel):
    resume_text: str
    job_description: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class JobMatchResponse(BaseModel):
    success: bool
    score: Optional[float] = None
    matched_skills: list[str] = []
    missing_skills: list[str] = []
    feedback: Optional[str] = None
    error: Optional[str] = None


class JdGenerateRequest(BaseModel):
    title: str
    company: Optional[str] = None
    skills: list[str] = []
    experience_level: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class JdGenerateResponse(BaseModel):
    success: bool
    description: Optional[str] = None
    error: Optional[str] = None


class AtsScanRequest(BaseModel):
    resume_text: str
    job_description: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AtsScanResponse(BaseModel):
    success: bool
    score: Optional[float] = None
    keyword_matches: list[str] = []
    missing_keywords: list[str] = []
    suggestions: list[str] = []
    error: Optional[str] = None


class SkillGapRequest(BaseModel):
    target_role: str
    current_skills: list[str] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class SkillGapResponse(BaseModel):
    success: bool
    current_skills: list[str] = []
    required_skills: list[str] = []
    gaps: list[str] = []
    recommendations: list[str] = []
    error: Optional[str] = None


class CareerRoadmapRequest(BaseModel):
    target_role: str
    current_role: Optional[str] = None
    time_horizon: Optional[str] = "12 months"
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class CareerRoadmapResponse(BaseModel):
    success: bool
    roadmap: Optional[list[dict]] = None
    milestones: list[str] = []
    error: Optional[str] = None


class CoverLetterRequest(BaseModel):
    job_title: str
    company: str
    resume_text: Optional[str] = None
    tone: Optional[str] = "professional"
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class CoverLetterResponse(BaseModel):
    success: bool
    letter: Optional[str] = None
    error: Optional[str] = None


class InterviewPrepRequest(BaseModel):
    job_title: str
    company: Optional[str] = None
    interview_type: Optional[str] = "general"
    skills: list[str] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class InterviewPrepResponse(BaseModel):
    success: bool
    questions: list[dict] = []
    tips: list[str] = []
    error: Optional[str] = None


class SkillAssessmentRequest(BaseModel):
    skill: str
    level: Optional[str] = "intermediate"
    num_questions: int = 10
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class SkillAssessmentResponse(BaseModel):
    success: bool
    questions: list[dict] = []
    skill: Optional[str] = None
    error: Optional[str] = None


class RecruiterSearchRequest(BaseModel):
    criteria: str
    filters: Optional[dict] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class RecruiterSearchResponse(BaseModel):
    success: bool
    candidates: list[dict] = []
    total_count: int = 0
    query: Optional[str] = None
    error: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    reply: Optional[str] = None
    conversation_id: Optional[str] = None
    error: Optional[str] = None

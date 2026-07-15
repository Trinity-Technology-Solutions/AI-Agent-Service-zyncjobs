from pydantic import BaseModel
from typing import Optional


class ExecuteRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_role: str = "candidate"
    context: Optional[dict] = None
    file_content: Optional[str] = None
    file_type: Optional[str] = None


class ExecuteResponse(BaseModel):
    success: bool
    intent: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    metadata: dict = {}


class AuthRequest(BaseModel):
    user_id: str
    role: str = "candidate"
    email: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    ollama: str = "unknown"


class VersionResponse(BaseModel):
    name: str
    version: str
    brains: list[str] = []
    knowledge_chunks: int = 0
    llm_provider: str = ""

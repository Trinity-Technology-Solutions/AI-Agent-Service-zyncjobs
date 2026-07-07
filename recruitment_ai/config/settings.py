import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "ZyncJobs AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    @validator('DEBUG', pre=True)
    def parse_debug(cls, v):
        if isinstance(v, bool):
            return v
        return str(v).lower() in ('1', 'true', 'yes', 'on')

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001

    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_MODELS: dict = {
        "chatbot": "llama3.1:8b",
        "job_parser": "qwen3:8b",
        "jd_generator": "llama3.1:8b",
        "ats_scanner": "qwen3:8b",
        "job_matching": "qwen3:8b",
        "career": "llama3.1:8b",
        "embedding": "nomic-embed-text",
    }

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "recruitment_ai")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "recruitment")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "recruitment_pass")

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_GRPC_PORT: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    QDRANT_COLLECTION: str = "zyncjobs_knowledge"

    @property
    def QDRANT_URL(self) -> str:
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    CHROMADB_PATH: str = "./chroma_db"
    CHROMADB_COLLECTION: str = "zyncjobs"

    RAG_TOP_K: int = 5
    RAG_RERANK_TOP_K: int = 3
    RAG_MAX_DISTANCE: float = 0.5
    RAG_FALLBACK_DISTANCE: float = 0.7
    RAG_MAX_CONTEXT_CHARS: int = 1500

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
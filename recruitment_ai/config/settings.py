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

    # LLM Provider: "ollama" (dev) or "bedrock" (production)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama (development)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))

    # Amazon Bedrock (production)
    AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")
    BEDROCK_MODEL: str = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0")
    BEDROCK_EMBED_MODEL: str = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    # Production model map — matches architecture doc Brain-to-Model table.
    # OllamaService reads this directly via settings.OLLAMA_MODELS.
    # Keys must match brain_name strings passed to ollama_service.generate().
    # Change values here to swap models without touching any brain code.
    # Model map — override via OLLAMA_MODELS env var in production.
    # Local dev defaults to qwen2.5:3b (only model currently pulled).
    # Production: pull llama3.1:8b and qwen3:8b then set via env.
    _DEFAULT_MODEL: str = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")

    OLLAMA_MODELS: dict = {
        # Candidate brains
        "career_advice": os.getenv("MODEL_CAREER_ADVICE", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "skill_assessment": os.getenv("MODEL_SKILL_ASSESSMENT", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "interview_prep": os.getenv("MODEL_INTERVIEW_PREP", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "resume_builder": os.getenv("MODEL_RESUME_BUILDER", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "resume_parser": os.getenv("MODEL_RESUME_PARSER", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "ats_scanner": os.getenv("MODEL_ATS_SCANNER", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "job_matching": os.getenv("MODEL_JOB_MATCHING", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        # Employer brains
        "job_parser": os.getenv("MODEL_JOB_PARSER", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "jd_generator": os.getenv("MODEL_JD_GENERATOR", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "recruiter": os.getenv("MODEL_RECRUITER", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        # Chatbot
        "chatbot": os.getenv("MODEL_CHATBOT", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        # Additional models from architecture doc
        "mistral": os.getenv("MODEL_MISTRAL", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        "gemma": os.getenv("MODEL_GEMMA", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        # Resume edit brain (section-specific AI edits)
        "resume_edit": os.getenv("MODEL_RESUME_EDIT", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        # Cover letter brain
        "cover_letter": os.getenv("MODEL_COVER_LETTER", os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:3b")),
        # Embeddings
        "embedding": "nomic-embed-text",
    }

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "zyncjobs_ai")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "zyncjobs")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "zyncjobs_pass")

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

    # AWS S3 — file storage for resumes, JD documents
    S3_BUCKET: str = os.getenv("S3_BUCKET", "zyncjobs.com")
    S3_REGION: str = os.getenv("S3_REGION", os.getenv("AWS_REGION", "ap-south-1"))
    S3_PREFIX: str = os.getenv("S3_PREFIX", "ai-service/")
    # AWS Textract — OCR for scanned PDFs and images (falls back to pdf-parse)
    TEXTRACT_ENABLED: bool = os.getenv("TEXTRACT_ENABLED", "true").lower() in ("1", "true", "yes")
    TEXTRACT_ROLE_ARN: str = os.getenv("TEXTRACT_ROLE_ARN", "")

    # Backend integration — calls ZyncJobs Node backend for data
    BACKEND_API_URL: str = os.getenv("BACKEND_API_URL", "http://localhost:5000")
    BACKEND_TIMEOUT: int = int(os.getenv("BACKEND_TIMEOUT", "30"))

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOKI_URL: str = os.getenv("LOKI_URL", "http://localhost:3100")

    def configure_logging(self):
        """Configure structured JSON logging + Loki handler if available."""
        import logging
        import sys
        logging.basicConfig(
            level=getattr(logging, self.LOG_LEVEL, logging.INFO),
            format='{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "msg": "%(message)s"}',
            stream=sys.stdout,
        )
        if self.LOKI_URL and self.LOKI_URL != "http://localhost:3100":
            try:
                import logging_loki  # python-logging-loki package
                loki_handler = logging_loki.LokiHandler(
                    url=f"{self.LOKI_URL}/loki/api/v1/push",
                    tags={"app": self.APP_NAME, "env": self.ENVIRONMENT},
                    version="1",
                )
                logging.getLogger().addHandler(loki_handler)
                logging.getLogger(__name__).info("Loki logging enabled: %s", self.LOKI_URL)
            except Exception as e:
                logging.getLogger(__name__).warning("Loki handler unavailable: %s", e)

    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in ("1", "true", "yes")

    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,https://zyncjobs.com,https://app.zyncjobs.com").split(",")

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = str(Path(__file__).resolve().parent.parent.parent / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_ENV_FILE, ".env"),
        case_sensitive=True,
        extra="ignore",
    )

    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8010

    AI_PROVIDER: str = "lmstudio"

    # -- LM Studio (local development) --
    LMSTUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    LMSTUDIO_MODEL: str = "gemma-3-4b-it"
    LMSTUDIO_TIMEOUT_SECONDS: int = 180
    LMSTUDIO_MAX_TOKENS: int = 800
    LMSTUDIO_TEMPERATURE: float = 0.3
    # Maximum characters of transcript to include in the LLM prompt.
    # Sized to fit within typical model context windows (4096 input tokens ≈ ~16 000 chars).
    # LM Studio small models (3–7B): set TRANSCRIPT_MAX_CHARS=8000 (conservative for 4K context).
    # Bedrock Nova Lite (128K context): can handle up to 40 000+ chars.
    # Set to 0 to disable the limit (only safe for very large context models).
    # The complete raw transcript is always preserved in the candidate — this limit
    # only controls what is forwarded to the LLM prompt.
    TRANSCRIPT_MAX_CHARS: int = 8000

    # -- AWS Bedrock (production) --
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None
    AWS_REGION: str = "ap-south-1"
    BEDROCK_MODEL_ID: str = "global.amazon.nova-2-lite-v1:0"
    BEDROCK_TIMEOUT_SECONDS: int = 60
    BEDROCK_MAX_TOKENS: int = 1024
    BEDROCK_TEMPERATURE: float = 0.3

    # -- PostgreSQL (shared DB) --
    DATABASE_URL: str = ""

    # -- Dashboard historical-data API --
    NAMBIKKAI_API_URL: str = ""
    AI_AGENT_API_KEY: str = ""
    NAMBIKKAI_API_TIMEOUT_SECONDS: int = 120
    NAMBIKKAI_API_PAGE_SIZE: int = 500

    # -- Email reporting (disabled by default) --
    EMAIL_ENABLED: bool = False
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    OWNER_EMAIL: str = ""

    # -- Bulk scan LLM batching & concurrency --
    LLM_BATCH_SIZE: int = 1000
    LLM_CONCURRENCY: int = 3

    # -- LLM provider fallback --
    LLM_FALLBACK_PROVIDER: str = ""

    # -- Bulk scan scheduler --
    AI_SCAN_INTERVAL_SECONDS: int = 3600

    # -- CORS allowed origins --
    CORS_ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:4000"

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> "Settings":
        provider = self.AI_PROVIDER.lower()
        if provider == "bedrock":
            if not self.AWS_REGION or not self.AWS_REGION.strip():
                raise ValueError("AWS_REGION must be configured when AI_PROVIDER=bedrock")
            if not self.BEDROCK_MODEL_ID or not self.BEDROCK_MODEL_ID.strip():
                raise ValueError("BEDROCK_MODEL_ID must be configured when AI_PROVIDER=bedrock")
        elif provider == "lmstudio":
            if not self.LMSTUDIO_BASE_URL or not self.LMSTUDIO_BASE_URL.strip():
                raise ValueError("LMSTUDIO_BASE_URL must be configured when AI_PROVIDER=lmstudio")
            if not self.LMSTUDIO_MODEL or not self.LMSTUDIO_MODEL.strip():
                raise ValueError("LMSTUDIO_MODEL must be configured when AI_PROVIDER=lmstudio")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

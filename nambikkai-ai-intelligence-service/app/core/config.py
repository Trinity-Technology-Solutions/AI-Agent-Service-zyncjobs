from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8010

    AI_PROVIDER: str = "lmstudio"

    # ── LM Studio (local development) ──────────────────────────────────
    LMSTUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    LMSTUDIO_MODEL: str = "gemma-3-4b-it"
    LMSTUDIO_TIMEOUT_SECONDS: int = 180
    LMSTUDIO_MAX_TOKENS: int = 800
    LMSTUDIO_TEMPERATURE: float = 0.3

    # ── AWS Bedrock (production) ────────────────────────────────────────
    AWS_REGION: str = "ap-south-1"
    BEDROCK_MODEL_ID: str = "amazon.nova-lite-v1:0"
    BEDROCK_MAX_TOKENS: int = 1024
    BEDROCK_TEMPERATURE: float = 0.3

    # ── Deterministic gating thresholds ────────────────────────────────
    VELOCITY_NOMINAL_THRESHOLD: float = 1.5
    VELOCITY_SURGE_THRESHOLD: float = 3.0
    # Reserved for future like-acceleration gating rule.
    LIKE_ACCELERATION_THRESHOLD: float = 5.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

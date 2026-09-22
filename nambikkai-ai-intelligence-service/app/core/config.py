from functools import lru_cache
from typing import Optional
from pydantic import model_validator
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

    # -- LM Studio (local development) --
    LMSTUDIO_BASE_URL: str = "http://127.0.0.1:1234/v1"
    LMSTUDIO_MODEL: str = "gemma-3-4b-it"
    LMSTUDIO_TIMEOUT_SECONDS: int = 180
    LMSTUDIO_MAX_TOKENS: int = 800
    LMSTUDIO_TEMPERATURE: float = 0.3

    # -- AWS Bedrock (production) --
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None
    AWS_REGION: str = "ap-south-1"
    BEDROCK_MODEL_ID: str = "global.amazon.nova-2-lite-v1:0"
    BEDROCK_TIMEOUT_SECONDS: int = 60
    BEDROCK_MAX_TOKENS: int = 1024
    BEDROCK_TEMPERATURE: float = 0.3

    # -- PostgreSQL (shared DB, AI-specific tables only) --
    DATABASE_URL: str = ""

    # -- Dashboard historical-data API --
    NAMBIKKAI_API_URL: str = ""
    AI_AGENT_API_KEY: str = ""
    NAMBIKKAI_API_TIMEOUT_SECONDS: int = 120
    NAMBIKKAI_API_PAGE_SIZE: int = 500

    # -- Deterministic gating thresholds (R&D spec) --
    # Vr < 1.5  -> NOMINAL
    # 1.5 <= Vr < 2.5 -> ELEVATED
    # Vr >= 2.5 OR La >= 8% -> BOOMING_SURGE
    VELOCITY_NOMINAL_THRESHOLD: float = 1.5
    VELOCITY_SURGE_THRESHOLD: float = 2.5
    LIKE_ACCELERATION_THRESHOLD: float = 8.0
    # velocity_ratio < threshold AND sufficient baseline -> LOW_PERFORMING
    LOW_PERFORMING_VELOCITY_THRESHOLD: float = 0.3

    # -- Email reporting (all disabled by default) --
    EMAIL_ENABLED: bool = False
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    OWNER_EMAIL: str = ""

    # -- Bulk scan LLM batching --
    # Max NEW LLM calls per platform per scan cycle.
    # Default is 1000 to ensure every eligible item is processed in one scan.
    # Every eligible item in a scan cycle MUST end as generated, unavailable,
    # failed_validation, or cached — never left as pending.
    LLM_BATCH_SIZE: int = 1000

    # -- Bounded LLM concurrency --
    # Number of concurrent LLM requests per scan cycle.
    # Tune separately for development (LM Studio, low concurrency) and production
    # (Bedrock, higher concurrency). Setting too high causes provider saturation.
    # Default 3 is safe for LM Studio; production Bedrock can handle higher values.
    LLM_CONCURRENCY: int = 3

    # -- LLM provider fallback --
    # Optional fallback provider to use when the primary provider is unavailable.
    # Leave blank (default) to disable fallback entirely.
    # Example: if AI_PROVIDER=bedrock fails, set LLM_FALLBACK_PROVIDER=lmstudio
    # WARNING: do NOT set fallback from production Bedrock to a local LM Studio
    # that is not reachable from the production server.
    LLM_FALLBACK_PROVIDER: str = ""

    # -- Bulk scan scheduler --
    AI_SCAN_INTERVAL_SECONDS: int = 3600

    # -- XGBoost model artifact --
    # Path to the directory containing xgboost_model.json and metadata.json.
    # Empty string means XGBoost loading is disabled (safe default).
    # Set to "models/track_a_xgboost" (or an absolute path) to enable.
    XGBOOST_MODEL_PATH: str = ""

    # -- XGBoost prediction quality gates --
    # Due to extreme class imbalance (surge events are rare, ~0.9% positive rate),
    # F1 alone is not a meaningful qualification gate. ROC-AUC is the primary gate
    # because it measures the model's discriminative ability independently of threshold.
    # A model with AUC >= 0.70 and recall >= 0.30 on the test set is genuinely useful
    # as an additional probabilistic signal alongside deterministic classification.
    #
    # XGBOOST_MIN_TEST_ROC_AUC: primary gate — model must beat random (0.5) by a wide margin
    # XGBOOST_MIN_TEST_RECALL:  secondary gate — model must catch a meaningful fraction of surges
    # XGBOOST_MIN_TEST_F1:      kept for backward compatibility; set low to reflect imbalance reality
    XGBOOST_MIN_TEST_F1: float = 0.03       # F1 >= 0.03 at 0.9% positive rate is meaningful
    XGBOOST_MIN_TEST_ROC_AUC: float = 0.70  # Primary gate: AUC must beat naive baseline clearly
    XGBOOST_MIN_TEST_RECALL: float = 0.30   # Must catch at least 30% of actual surges

    # -- CORS allowed origins --
    # Comma-separated list of allowed origins for the AI service.
    # Default allows localhost development. In production set to your
    # dashboard origin, e.g. "https://nambikkai.info,https://www.nambikkai.info"
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

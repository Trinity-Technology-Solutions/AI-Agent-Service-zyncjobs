import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "ZyncJobs AI Service")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Ollama model parameters
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.3"))
    OLLAMA_NUM_CTX: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    OLLAMA_NUM_PREDICT: int = int(os.getenv("OLLAMA_NUM_PREDICT", "300"))
    OLLAMA_TOP_P: float = float(os.getenv("OLLAMA_TOP_P", "0.85"))
    OLLAMA_TOP_K: int = int(os.getenv("OLLAMA_TOP_K", "30"))
    OLLAMA_REPEAT_PENALTY: float = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.0"))
    OLLAMA_NUM_THREAD: int = int(os.getenv("OLLAMA_NUM_THREAD", "8"))
    OLLAMA_NUM_GPU: int = int(os.getenv("OLLAMA_NUM_GPU", "0"))
    OLLAMA_SEED: int = int(os.getenv("OLLAMA_SEED", "0"))
    OLLAMA_STOP: list[str] = os.getenv("OLLAMA_STOP", "").split(",") if os.getenv("OLLAMA_STOP") else []
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "60"))

    # Retry
    OLLAMA_RETRY_MAX: int = int(os.getenv("OLLAMA_RETRY_MAX", "3"))
    OLLAMA_RETRY_DELAY: float = float(os.getenv("OLLAMA_RETRY_DELAY", "1.0"))

    # Agent parameters
    AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "3"))
    AGENT_MEMORY_WINDOW: int = int(os.getenv("AGENT_MEMORY_WINDOW", "10"))

    # Redis (optional — fallback to in-memory if not set)
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    CONVERSATION_TTL: int = int(os.getenv("CONVERSATION_TTL", "3600"))

    @property
    def OLLAMA_OPTIONS(self) -> dict:
        opts = {
            "temperature": self.OLLAMA_TEMPERATURE,
            "num_ctx": self.OLLAMA_NUM_CTX,
            "num_predict": self.OLLAMA_NUM_PREDICT,
            "top_p": self.OLLAMA_TOP_P,
            "top_k": self.OLLAMA_TOP_K,
            "repeat_penalty": self.OLLAMA_REPEAT_PENALTY,
            "num_thread": self.OLLAMA_NUM_THREAD,
        }
        if self.OLLAMA_NUM_GPU > 0:
            opts["num_gpu"] = self.OLLAMA_NUM_GPU
        if self.OLLAMA_SEED > 0:
            opts["seed"] = self.OLLAMA_SEED
        return opts


settings = Settings()

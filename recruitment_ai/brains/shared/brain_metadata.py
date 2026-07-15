"""ExecutionMetadata — typed metadata for every brain execution.
Tracks provider, model, tokens, cost, latency, and fallback status.
Enabled by Logging phase, but the type exists here so all brains return consistent metadata.
"""
from pydantic import BaseModel


class ExecutionMetadata(BaseModel):
    """Structured execution metadata collected by MasterBrain after each brain.run()."""
    brain_name: str = ""
    intent: str = ""
    provider: str = ""
    model: str = ""
    execution_time_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    cache_hit: bool = False
    rag_used: bool = False
    fallback: bool = False
    error: str | None = None

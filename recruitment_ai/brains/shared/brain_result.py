"""Enterprise BrainResult — every Brain.run() returns this.
Architecture doc: Brain → Validator → Frontend.

Extended in Phase 7b with:
  - response (replaces data)
  - citations for source attribution
  - execution_time, tokens, cost for observability
  - warnings for non-blocking issues
"""
from pydantic import BaseModel


class BrainResult(BaseModel):
    """Structured result from a brain execution.

    Fields:
        success: True if execution completed without errors.
        response: The brain's output payload (reply text, JSON, etc.).
        citations: Source attributions for RAG responses.
        metadata: Arbitrary execution metadata.
        execution_time: Wall-clock time in seconds.
        tokens: Total tokens consumed.
        cost: Estimated cost in USD.
        warnings: Non-blocking issues found during execution.
    """
    success: bool = True
    response: dict | None = None
    error: str | None = None
    citations: list[dict] = []
    metadata: dict = {}
    execution_time: float = 0.0
    tokens: int = 0
    cost: float = 0.0
    warnings: list[str] = []

    # ── Backward compat ────────────────────────────────────────────────────
    @property
    def data(self) -> dict | None:
        return self.response

    @data.setter
    def data(self, value: dict | None) -> None:
        self.response = value

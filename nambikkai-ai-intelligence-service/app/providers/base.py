from abc import ABC, abstractmethod
from typing import Any, Optional
from app.domain.models import EvidencePackage, EditorialAnalysis


class LLMProvider(ABC):

    @abstractmethod
    async def generate_structured_analysis(
        self,
        evidence: EvidencePackage,
        feedback: Optional[str] = None,
    ) -> EditorialAnalysis:
        """Generate editorial analysis from a bounded evidence package, with optional correction feedback."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""

    @abstractmethod
    def get_provider_metadata(self) -> dict[str, Any]:
        """Return provider name and configuration (no secrets)."""

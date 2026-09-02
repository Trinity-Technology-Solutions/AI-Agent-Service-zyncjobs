from abc import ABC, abstractmethod
from typing import Any
from app.domain.models import EvidencePackage, EditorialAnalysis


class LLMProvider(ABC):

    @abstractmethod
    async def generate_structured_analysis(self, evidence: EvidencePackage) -> EditorialAnalysis:
        """Generate editorial analysis from a bounded evidence package."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""

    @abstractmethod
    def get_provider_metadata(self) -> dict[str, Any]:
        """Return provider name and configuration (no secrets)."""

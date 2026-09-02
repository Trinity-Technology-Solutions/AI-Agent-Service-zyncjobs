from abc import ABC, abstractmethod
from app.domain.models import ContentMetadata, ContentMetrics


class AnalyticsDataSource(ABC):
    """
    Abstract boundary for the Nambikkai analytics data layer.

    The real implementation will be added in the next phase once the
    actual JSON response schema from the analytics API is confirmed.
    Do not invent field names here.
    """

    @abstractmethod
    async def fetch_latest_metrics(self, content_id: str) -> ContentMetrics:
        """Fetch the most recent metrics for a content item."""

    @abstractmethod
    async def fetch_content_metadata(self, content_id: str) -> ContentMetadata:
        """Fetch metadata for a content item."""

    @abstractmethod
    async def fetch_content_history(self, content_id: str, hours: int = 168) -> list[dict]:
        """Fetch raw historical data points. Schema TBD."""

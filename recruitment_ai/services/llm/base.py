"""Abstract LLM provider interface."""
from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):

    @abstractmethod
    async def generate(
        self,
        brain_name: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        pass

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    async def close(self):
        pass

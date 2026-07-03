from abc import ABC, abstractmethod
from typing import Optional, Generator


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, system: Optional[str] = None, **kwargs):
        pass

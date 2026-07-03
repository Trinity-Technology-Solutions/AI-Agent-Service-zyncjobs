from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @property
    def result_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass

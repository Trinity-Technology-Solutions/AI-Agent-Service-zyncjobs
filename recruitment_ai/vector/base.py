from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    id: str
    text: str
    source: str = "unknown"
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

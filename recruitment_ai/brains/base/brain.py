"""Brain abstract base class — every brain implements this contract.
Architecture doc: IntentClassifier → BrainRouter → Brain.run(state) → BrainResult.
"""
from abc import ABC, abstractmethod
from recruitment_ai.brains.shared.brain_state import BrainState
from recruitment_ai.brains.shared.brain_result import BrainResult


class Brain(ABC):
    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    async def run(self, state: BrainState) -> BrainResult:
        ...

    async def validate(self, state: BrainState) -> bool:
        return True

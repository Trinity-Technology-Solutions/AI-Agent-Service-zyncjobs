"""BrainRegistry — centralized registry mapping intent → Brain instance.
"""
from recruitment_ai.brains.base.brain import Brain


class BrainRegistry:
    def __init__(self):
        self._registry: dict[str, Brain] = {}

    def register(self, intent: str, brain: Brain) -> None:
        self._registry[intent] = brain

    def register_many(self, intents: list[str], brain: Brain) -> None:
        for intent in intents:
            self._registry[intent] = brain

    def get(self, intent: str) -> Brain | None:
        return self._registry.get(intent)

    def unregister(self, intent: str) -> None:
        self._registry.pop(intent, None)

    @property
    def intents(self) -> list[str]:
        return list(self._registry.keys())

    @property
    def all_brains(self) -> dict[str, Brain]:
        return dict(self._registry)

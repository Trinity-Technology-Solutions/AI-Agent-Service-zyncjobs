import asyncio
from abc import ABC, abstractmethod
from typing import Optional
from app.llm.router import router as llm_router
from app.knowledge.knowledge_base import knowledge_base
from app.utils.logger import logger


class BaseAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.llm = llm_router

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    async def run(self, query: str, user_id: Optional[str] = None, **kwargs) -> dict:
        logger.info(f"Agent {self.name} processing query")
        return await self.execute(query, user_id=user_id, **kwargs)

    @abstractmethod
    async def execute(self, query: str, user_id: Optional[str] = None, **kwargs) -> dict:
        pass

    async def generate(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        return await asyncio.to_thread(
            self.llm.generate, prompt, system=system or self.system_prompt(), **kwargs
        )

    def retrieve_context(self, query: str, max_chars: int = 1500) -> str:
        context = knowledge_base.build_context(query, max_chars=max_chars)
        if context:
            logger.info(f"RAG context retrieved ({len(context)} chars)")
        return context

    def augment_with_context(self, prompt: str, query: str) -> str:
        context = self.retrieve_context(query)
        if context:
            return f"{prompt}\n\nUse the following knowledge to inform your response, but DO NOT include it verbatim in your output:\n{context}"
        return prompt

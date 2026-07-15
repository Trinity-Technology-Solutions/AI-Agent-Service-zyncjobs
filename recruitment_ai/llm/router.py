"""LLM Router — single entry point for all AI calls.
Matches roadmap spec:
  response = router.generate(provider="bedrock", model="claude", prompt=prompt)
"""
from typing import Optional


class LLMRouter:

    async def generate(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        prompt: str = "",
        brain_name: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        if provider:
            return await self._generate_with_provider(provider, model or brain_name or "default", prompt, system, temperature, max_tokens)
        from recruitment_ai.services.llm.factory import get_llm
        return await get_llm().generate(brain_name or "chatbot", prompt, system, temperature, max_tokens)

    async def _generate_with_provider(self, provider: str, model: str, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> str:
        if provider.lower() == "bedrock":
            from recruitment_ai.llm.bedrock import BedrockProvider
            llm = BedrockProvider()
        else:
            from recruitment_ai.llm.ollama import OllamaProvider
            llm = OllamaProvider()
        return await llm.generate(model, prompt, system, temperature, max_tokens)

    async def embed(self, text: str) -> list[float]:
        from recruitment_ai.services.llm.factory import get_llm
        return await get_llm().embed(text)

    async def health_check(self) -> bool:
        from recruitment_ai.services.llm.factory import get_llm
        return await get_llm().health_check()

    async def close(self):
        from recruitment_ai.services.llm.factory import get_llm
        await get_llm().close()


class LLMService:
    """Backward-compat delegate — matches original shared.llm_service.LLMService interface."""

    async def generate(self, brain_name: str, prompt: str, system: Optional[str] = None, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        from recruitment_ai.llm import llm_router
        return await llm_router.generate(brain_name=brain_name, prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens)

    async def embed(self, text: str) -> list[float]:
        from recruitment_ai.llm import llm_router
        return await llm_router.embed(text)

    async def health_check(self) -> bool:
        from recruitment_ai.llm import llm_router
        return await llm_router.health_check()

    async def close(self):
        from recruitment_ai.llm import llm_router
        await llm_router.close()

"""Amazon Bedrock LLM provider — used in production."""
import json
from typing import Optional
from recruitment_ai.services.llm.base import BaseLLMProvider
from recruitment_ai.config.settings import settings


class BedrockProvider(BaseLLMProvider):

    def __init__(self):
        import boto3
        self._client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
        self._embed_client = boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)
        self.model_id = settings.BEDROCK_MODEL

    async def generate(
        self,
        brain_name: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_generate, prompt, system, temperature, max_tokens
        )

    def _sync_generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        body: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "anthropic_version": "bedrock-2023-05-31",
        }
        if system:
            body["system"] = system

        resp = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        return result.get("content", [{}])[0].get("text", "")

    async def embed(self, text: str) -> list[float]:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_embed, text
        )

    def _sync_embed(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text})
        resp = self._embed_client.invoke_model(
            modelId=settings.BEDROCK_EMBED_MODEL,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(resp["body"].read())
        return result.get("embedding", [])

    async def health_check(self) -> bool:
        try:
            import boto3
            client = boto3.client("bedrock", region_name=settings.AWS_REGION)
            client.list_foundation_models(byOutputModality="TEXT")
            return True
        except Exception:
            return False

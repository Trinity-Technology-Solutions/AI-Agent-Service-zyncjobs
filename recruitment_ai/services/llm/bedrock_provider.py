import logging
from typing import Optional
from recruitment_ai.services.llm.base import BaseLLMProvider
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)


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
            None, self._sync_generate, brain_name, prompt, system, temperature, max_tokens
        )

    def _sync_generate(
        self,
        brain_name: str,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        logger.info(
            "Calling Bedrock | model=%s | brain=%s",
            self.model_id,
            brain_name,
        )
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]

        resp = self._client.converse(**kwargs)
        return resp["output"]["message"]["content"][0]["text"]

    async def embed(self, text: str) -> list[float]:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, self._sync_embed, text
        )

    def _sync_embed(self, text: str) -> list[float]:
        import json
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

    async def close(self):
        pass

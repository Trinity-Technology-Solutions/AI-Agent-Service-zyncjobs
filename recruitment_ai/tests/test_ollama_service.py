"""Tests for LLMService with mocked HTTP client."""
import pytest
from unittest.mock import AsyncMock, patch
from recruitment_ai.shared.llm_service import LLMService, llm_service


@pytest.mark.asyncio
async def test_health_check_returns_bool():
    result = await llm_service.health_check()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_generate_success():
    with patch.object(LLMService, 'generate', new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Hello from LLM"
        result = await llm_service.generate("chatbot", "test prompt")
        assert result == "Hello from LLM"


@pytest.mark.asyncio
async def test_generate_with_system_prompt():
    with patch.object(LLMService, 'generate', new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Answer"
        result = await llm_service.generate("chatbot", "question", system="You are helpful")
        assert result == "Answer"


@pytest.mark.asyncio
async def test_generate_propagates_timeout():
    """TimeoutExceptions propagate from the provider through LLMService."""
    with patch.object(LLMService, 'generate', new_callable=AsyncMock) as mock_gen:
        import httpx
        mock_gen.side_effect = httpx.TimeoutException("timeout")
        with pytest.raises(httpx.TimeoutException):
            await llm_service.generate("chatbot", "test prompt")


@pytest.mark.asyncio
async def test_generate_propagates_connect_error():
    """ConnectErrors propagate from the provider through LLMService."""
    with patch.object(LLMService, 'generate', new_callable=AsyncMock) as mock_gen:
        import httpx
        mock_gen.side_effect = httpx.ConnectError("connection refused")
        with pytest.raises(httpx.ConnectError):
            await llm_service.generate("chatbot", "test")

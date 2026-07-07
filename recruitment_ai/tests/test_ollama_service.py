"""Tests for OllamaService with mocked HTTP client."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from recruitment_ai.shared.ollama_service import OllamaService, ollama_service


@pytest.mark.asyncio
async def test_get_model_returns_configured_model():
    model = ollama_service.get_model("chatbot")
    assert model is not None
    assert isinstance(model, str)


@pytest.mark.asyncio
async def test_get_model_returns_default_for_unknown():
    model = ollama_service.get_model("nonexistent_brain")
    assert model is not None


@pytest.mark.asyncio
async def test_model_map_has_all_brains():
    expected_keys = {"chatbot", "job_parser", "jd_generator", "resume_parser",
                     "ats_scanner", "job_matching", "career_advice",
                     "skill_assessment", "interview_prep", "resume_builder",
                     "recruiter"}
    for key in expected_keys:
        assert key in ollama_service.MODEL_MAP, f"{key} missing from MODEL_MAP"


@pytest.mark.asyncio
async def test_health_check_returns_bool():
    result = await ollama_service.health_check()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_generate_fallback_on_timeout():
    """Test that generate returns empty string after retries fail."""
    with patch.object(ollama_service, '_do_generate', new_callable=AsyncMock) as mock_gen:
        import httpx
        mock_gen.side_effect = httpx.TimeoutException("timeout")
        result = await ollama_service.generate("chatbot", "test prompt")
        assert result == ""


@pytest.mark.asyncio
async def test_generate_success():
    with patch.object(ollama_service, '_do_generate', new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Hello from Ollama"
        result = await ollama_service.generate("chatbot", "test prompt")
        assert result == "Hello from Ollama"


@pytest.mark.asyncio
async def test_generate_with_system_prompt():
    with patch.object(ollama_service, '_do_generate', new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = "Answer"
        result = await ollama_service.generate("chatbot", "question", system="You are helpful")
        assert result == "Answer"


@pytest.mark.asyncio
async def test_generate_retries_on_connect_error():
    """Test all retries exhausted returns empty string."""
    with patch.object(ollama_service, '_do_generate', new_callable=AsyncMock) as mock_gen:
        import httpx
        mock_gen.side_effect = httpx.ConnectError("connection refused")
        result = await ollama_service.generate("chatbot", "test")
        assert result == ""


@pytest.mark.asyncio
async def test_do_generate_sends_correct_payload():
    """Verify the payload sent to Ollama has the right structure."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "response text"}}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    service = OllamaService()
    with patch.object(service, '_get_client', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_client
        result = await service._do_generate("qwen2.5:3b", [{"role": "user", "content": "hi"}])
        assert result == "response text"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args[1]
        assert call_args["json"]["model"] == "qwen2.5:3b"
        assert call_args["json"]["stream"] is False


@pytest.mark.asyncio
async def test_close_cleans_up():
    mock_client = AsyncMock()
    mock_client.is_closed = False
    mock_client.aclose = AsyncMock()
    original = ollama_service._client
    ollama_service._client = mock_client
    await ollama_service.close()
    mock_client.aclose.assert_awaited_once()
    ollama_service._client = original

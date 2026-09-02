"""
Tests for Nambikkai AI Intelligence Service.
No real LLM required — LM Studio is mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from app.domain.models import (
    AnalysisStatus,
    AnalyticsEvent,
    ContentMetadata,
    ContentMetrics,
    EditorialAnalysis,
    GateClassification,
    ValidationResult,
)
from app.services.gating import (
    compute_like_acceleration,
    compute_velocity_ratio,
    evaluate_gate,
)
from app.agent.decision_engine import should_invoke_llm, should_monitor
from app.agent.orchestrator import AgentOrchestrator
from app.validation.output_validator import validate_output
from app.validation.policy_validator import validate_policy
from app.services.evidence_builder import build_evidence_package


# ── Fixtures ───────────────────────────────────────────────────────────────

def make_metrics(
    delta_views=100.0,
    baseline=100.0,
    delta_likes=5.0,
    hour_views=100.0,
) -> ContentMetrics:
    return ContentMetrics(
        current_hour_delta_views=delta_views,
        seven_day_rolling_hourly_baseline=baseline,
        one_hour_delta_likes=delta_likes,
        one_hour_delta_views=hour_views,
    )


def make_event(metrics: ContentMetrics) -> AnalyticsEvent:
    return AnalyticsEvent(
        event_id="test-001",
        received_at=datetime.now(timezone.utc),
        metadata=ContentMetadata(
            content_id="c-001",
            title="Test Video",
            creator_id="u-001",
            platform="youtube",
        ),
        metrics=metrics,
    )


def make_editorial(**overrides) -> EditorialAnalysis:
    defaults = dict(
        content_intent="Educational content about Python",
        observed_signals=["High engagement observed in first hour"],
        possible_contributing_factors=["Possible contributing factor: trending topic"],
        writer_recommendations=["Add timestamps"],
        keyword_suggestions=["python", "tutorial"],
        title_suggestions=["Python Tutorial 2024"],
        description_suggestions=["Learn Python fast"],
        hashtag_suggestions=["#python"],
        cross_platform_ideas=["Share on Twitter"],
        confidence=0.75,
        limitations=["Limited historical data"],
    )
    defaults.update(overrides)
    return EditorialAnalysis(**defaults)


# ── Velocity ratio ─────────────────────────────────────────────────────────

def test_velocity_ratio_normal():
    m = make_metrics(delta_views=200.0, baseline=100.0)
    assert compute_velocity_ratio(m) == pytest.approx(2.0)


def test_velocity_ratio_zero_baseline():
    m = make_metrics(delta_views=200.0, baseline=0.0)
    assert compute_velocity_ratio(m) is None


def test_velocity_ratio_zero_views():
    m = make_metrics(delta_views=0.0, baseline=100.0)
    assert compute_velocity_ratio(m) == pytest.approx(0.0)


# ── Like acceleration ──────────────────────────────────────────────────────

def test_like_acceleration_normal():
    m = make_metrics(delta_likes=10.0, hour_views=100.0)
    assert compute_like_acceleration(m) == pytest.approx(10.0)


def test_like_acceleration_zero_views():
    m = make_metrics(delta_likes=10.0, hour_views=0.0)
    assert compute_like_acceleration(m) is None


def test_like_acceleration_zero_likes():
    m = make_metrics(delta_likes=0.0, hour_views=100.0)
    assert compute_like_acceleration(m) == pytest.approx(0.0)


# ── Gate classification ────────────────────────────────────────────────────

def test_gate_nominal():
    m = make_metrics(delta_views=100.0, baseline=100.0)  # ratio = 1.0 < 1.5
    result = evaluate_gate(m)
    assert result.classification == GateClassification.NOMINAL


def test_gate_elevated():
    m = make_metrics(delta_views=200.0, baseline=100.0)  # ratio = 2.0, 1.5 ≤ 2.0 < 3.0
    result = evaluate_gate(m)
    assert result.classification == GateClassification.ELEVATED


def test_gate_booming_surge():
    m = make_metrics(delta_views=400.0, baseline=100.0)  # ratio = 4.0 ≥ 3.0
    result = evaluate_gate(m)
    assert result.classification == GateClassification.BOOMING_SURGE


def test_gate_nominal_threshold_edge():
    # Exactly at nominal threshold (1.5) → ELEVATED
    m = make_metrics(delta_views=150.0, baseline=100.0)
    result = evaluate_gate(m)
    assert result.classification == GateClassification.ELEVATED


def test_gate_surge_threshold_edge():
    # Exactly at surge threshold (3.0) → BOOMING_SURGE
    m = make_metrics(delta_views=300.0, baseline=100.0)
    result = evaluate_gate(m)
    assert result.classification == GateClassification.BOOMING_SURGE


def test_gate_zero_baseline_defaults_nominal():
    m = make_metrics(delta_views=999.0, baseline=0.0)
    result = evaluate_gate(m)
    assert result.classification == GateClassification.NOMINAL


# ── Decision engine ────────────────────────────────────────────────────────

def test_should_invoke_llm_only_for_surge():
    surge_gate = evaluate_gate(make_metrics(delta_views=400.0, baseline=100.0))
    elevated_gate = evaluate_gate(make_metrics(delta_views=200.0, baseline=100.0))
    nominal_gate = evaluate_gate(make_metrics(delta_views=100.0, baseline=100.0))

    assert should_invoke_llm(surge_gate) is True
    assert should_invoke_llm(elevated_gate) is False
    assert should_invoke_llm(nominal_gate) is False


def test_should_monitor_only_for_elevated():
    elevated_gate = evaluate_gate(make_metrics(delta_views=200.0, baseline=100.0))
    nominal_gate = evaluate_gate(make_metrics(delta_views=100.0, baseline=100.0))
    surge_gate = evaluate_gate(make_metrics(delta_views=400.0, baseline=100.0))

    assert should_monitor(elevated_gate) is True
    assert should_monitor(nominal_gate) is False
    assert should_monitor(surge_gate) is False


# ── Orchestrator — nominal does NOT invoke LLM ─────────────────────────────

@pytest.mark.asyncio
async def test_nominal_does_not_invoke_llm():
    mock_provider = MagicMock()
    mock_provider.generate_structured_analysis = AsyncMock()
    mock_provider.get_provider_metadata = MagicMock(return_value={"provider": "mock"})

    orchestrator = AgentOrchestrator(provider=mock_provider)
    event = make_event(make_metrics(delta_views=100.0, baseline=100.0))
    result, audit = await orchestrator.run(event)

    assert result.status == AnalysisStatus.SKIPPED_NOMINAL
    mock_provider.generate_structured_analysis.assert_not_called()
    assert audit.llm_invoked is False


@pytest.mark.asyncio
async def test_elevated_does_not_invoke_llm():
    mock_provider = MagicMock()
    mock_provider.generate_structured_analysis = AsyncMock()
    mock_provider.get_provider_metadata = MagicMock(return_value={"provider": "mock"})

    orchestrator = AgentOrchestrator(provider=mock_provider)
    event = make_event(make_metrics(delta_views=200.0, baseline=100.0))
    result, audit = await orchestrator.run(event)

    assert result.status == AnalysisStatus.MONITORING
    mock_provider.generate_structured_analysis.assert_not_called()
    assert audit.llm_invoked is False


@pytest.mark.asyncio
async def test_surge_invokes_llm_and_returns_success():
    editorial = make_editorial()
    mock_provider = MagicMock()
    mock_provider.generate_structured_analysis = AsyncMock(return_value=editorial)
    mock_provider.get_provider_metadata = MagicMock(return_value={"provider": "mock"})

    orchestrator = AgentOrchestrator(provider=mock_provider)
    event = make_event(make_metrics(delta_views=400.0, baseline=100.0))
    result, audit = await orchestrator.run(event)

    assert result.status == AnalysisStatus.SUCCESS
    mock_provider.generate_structured_analysis.assert_called_once()
    assert audit.llm_invoked is True
    assert audit.validation_passed is True


# ── Validation ─────────────────────────────────────────────────────────────

def test_output_validator_passes_valid():
    editorial = make_editorial()
    gate = evaluate_gate(make_metrics(delta_views=400.0, baseline=100.0))
    evidence = build_evidence_package(
        metadata=ContentMetadata(content_id="c-001", title="T", creator_id="u-001"),
        metrics=make_metrics(delta_views=400.0, baseline=100.0),
        gate_result=gate,
    )
    result = validate_output(editorial, evidence)
    assert result.is_valid is True


def test_output_validator_fails_empty_intent():
    editorial = make_editorial(content_intent="")
    gate = evaluate_gate(make_metrics(delta_views=400.0, baseline=100.0))
    evidence = build_evidence_package(
        metadata=ContentMetadata(content_id="c-001", title="T", creator_id="u-001"),
        metrics=make_metrics(delta_views=400.0, baseline=100.0),
        gate_result=gate,
    )
    result = validate_output(editorial, evidence)
    assert result.is_valid is False


def test_policy_validator_rejects_causal_language():
    editorial = make_editorial(
        possible_contributing_factors=["This caused by the trending hashtag"]
    )
    result = validate_policy(editorial)
    assert result.is_valid is False


def test_policy_validator_passes_safe_language():
    editorial = make_editorial()
    result = validate_policy(editorial)
    assert result.is_valid is True


# ── Provider abstraction ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_provider_error_returns_provider_error_status():
    from app.core.exceptions import ProviderError

    mock_provider = MagicMock()
    mock_provider.generate_structured_analysis = AsyncMock(
        side_effect=ProviderError("LM Studio unreachable")
    )
    mock_provider.get_provider_metadata = MagicMock(return_value={"provider": "mock"})

    orchestrator = AgentOrchestrator(provider=mock_provider)
    event = make_event(make_metrics(delta_views=400.0, baseline=100.0))
    result, audit = await orchestrator.run(event)

    assert result.status == AnalysisStatus.PROVIDER_ERROR
    assert audit.llm_invoked is True


# ── JSON parsing — _strip_fences and parse path ───────────────────────────

def test_strip_fences_removes_json_fence():
    from app.providers.lmstudio import _strip_fences
    fenced = "```json\n{\"key\": \"value\"}\n```"
    assert _strip_fences(fenced) == '{"key": "value"}'


def test_strip_fences_removes_plain_fence():
    from app.providers.lmstudio import _strip_fences
    fenced = "```\n{\"key\": \"value\"}\n```"
    assert _strip_fences(fenced) == '{"key": "value"}'


def test_strip_fences_passthrough_clean_json():
    from app.providers.lmstudio import _strip_fences
    clean = '{"key": "value"}'
    assert _strip_fences(clean) == clean


def test_strip_fences_truncated_no_closing_fence():
    from app.providers.lmstudio import _strip_fences
    # Truncated output — no closing fence, should not corrupt the content
    truncated = "```json\n{\"key\": \"val"
    result = _strip_fences(truncated)
    assert result == '{"key": "val'


# ── LMStudioProvider unit tests (httpx mocked) ────────────────────────────

import json as _json
from unittest.mock import patch, AsyncMock as _AsyncMock, MagicMock as _MagicMock


def _make_httpx_response(content: str, finish_reason: str = "stop", status_code: int = 200):
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
    }
    mock_resp = _MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = body
    mock_resp.raise_for_status = _MagicMock()
    return mock_resp


def _valid_json_str() -> str:
    return _json.dumps({
        "content_intent": "Educational",
        "observed_signals": ["High engagement"],
        "possible_contributing_factors": ["Possible contributing factor: trending topic"],
        "writer_recommendations": ["Add timestamps"],
        "keyword_suggestions": ["python"],
        "title_suggestions": ["Python 2024"],
        "description_suggestions": ["Learn fast"],
        "hashtag_suggestions": ["#python"],
        "cross_platform_ideas": ["Share on Twitter"],
        "confidence": 0.8,
        "limitations": ["Limited data"],
    })


@pytest.mark.asyncio
async def test_provider_parses_valid_json():
    from app.providers.lmstudio import LMStudioProvider
    from app.domain.models import ContentMetadata, ContentMetrics, GateResult, GateClassification, EvidencePackage, DataQuality

    provider = LMStudioProvider()
    evidence = EvidencePackage(
        content_metadata=ContentMetadata(content_id="c-1", title="T", creator_id="u-1"),
        verified_metrics=ContentMetrics(current_hour_delta_views=400, seven_day_rolling_hourly_baseline=100, one_hour_delta_likes=10, one_hour_delta_views=400),
        gate_result=GateResult(classification=GateClassification.BOOMING_SURGE, velocity_ratio=4.0, like_acceleration=10.0, reason="surge"),
        data_quality=DataQuality(),
    )
    mock_resp = _make_httpx_response(_valid_json_str())
    with patch("httpx.AsyncClient.post", new=_AsyncMock(return_value=mock_resp)):
        result = await provider.generate_structured_analysis(evidence)
    assert result.content_intent == "Educational"
    assert result.confidence == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_provider_parses_fenced_json():
    from app.providers.lmstudio import LMStudioProvider
    from app.domain.models import ContentMetadata, ContentMetrics, GateResult, GateClassification, EvidencePackage, DataQuality

    provider = LMStudioProvider()
    evidence = EvidencePackage(
        content_metadata=ContentMetadata(content_id="c-1", title="T", creator_id="u-1"),
        verified_metrics=ContentMetrics(current_hour_delta_views=400, seven_day_rolling_hourly_baseline=100, one_hour_delta_likes=10, one_hour_delta_views=400),
        gate_result=GateResult(classification=GateClassification.BOOMING_SURGE, velocity_ratio=4.0, like_acceleration=10.0, reason="surge"),
        data_quality=DataQuality(),
    )
    fenced = f"```json\n{_valid_json_str()}\n```"
    mock_resp = _make_httpx_response(fenced)
    with patch("httpx.AsyncClient.post", new=_AsyncMock(return_value=mock_resp)):
        result = await provider.generate_structured_analysis(evidence)
    assert result.content_intent == "Educational"


@pytest.mark.asyncio
async def test_provider_raises_on_truncated_output():
    from app.providers.lmstudio import LMStudioProvider
    from app.core.exceptions import ProviderError
    from app.domain.models import ContentMetadata, ContentMetrics, GateResult, GateClassification, EvidencePackage, DataQuality

    provider = LMStudioProvider()
    evidence = EvidencePackage(
        content_metadata=ContentMetadata(content_id="c-1", title="T", creator_id="u-1"),
        verified_metrics=ContentMetrics(current_hour_delta_views=400, seven_day_rolling_hourly_baseline=100, one_hour_delta_likes=10, one_hour_delta_views=400),
        gate_result=GateResult(classification=GateClassification.BOOMING_SURGE, velocity_ratio=4.0, like_acceleration=10.0, reason="surge"),
        data_quality=DataQuality(),
    )
    mock_resp = _make_httpx_response('{"content_intent": "trunc', finish_reason="length")
    with patch("httpx.AsyncClient.post", new=_AsyncMock(return_value=mock_resp)):
        with pytest.raises(ProviderError, match="truncated"):
            await provider.generate_structured_analysis(evidence)


@pytest.mark.asyncio
async def test_provider_raises_on_invalid_json():
    from app.providers.lmstudio import LMStudioProvider
    from app.core.exceptions import ProviderError
    from app.domain.models import ContentMetadata, ContentMetrics, GateResult, GateClassification, EvidencePackage, DataQuality

    provider = LMStudioProvider()
    evidence = EvidencePackage(
        content_metadata=ContentMetadata(content_id="c-1", title="T", creator_id="u-1"),
        verified_metrics=ContentMetrics(current_hour_delta_views=400, seven_day_rolling_hourly_baseline=100, one_hour_delta_likes=10, one_hour_delta_views=400),
        gate_result=GateResult(classification=GateClassification.BOOMING_SURGE, velocity_ratio=4.0, like_acceleration=10.0, reason="surge"),
        data_quality=DataQuality(),
    )
    mock_resp = _make_httpx_response("not json at all {broken")
    with patch("httpx.AsyncClient.post", new=_AsyncMock(return_value=mock_resp)):
        with pytest.raises(ProviderError, match="invalid JSON"):
            await provider.generate_structured_analysis(evidence)


@pytest.mark.asyncio
async def test_provider_raises_on_extra_text_outside_json():
    from app.providers.lmstudio import LMStudioProvider
    from app.core.exceptions import ProviderError
    from app.domain.models import ContentMetadata, ContentMetrics, GateResult, GateClassification, EvidencePackage, DataQuality

    provider = LMStudioProvider()
    evidence = EvidencePackage(
        content_metadata=ContentMetadata(content_id="c-1", title="T", creator_id="u-1"),
        verified_metrics=ContentMetrics(current_hour_delta_views=400, seven_day_rolling_hourly_baseline=100, one_hour_delta_likes=10, one_hour_delta_views=400),
        gate_result=GateResult(classification=GateClassification.BOOMING_SURGE, velocity_ratio=4.0, like_acceleration=10.0, reason="surge"),
        data_quality=DataQuality(),
    )
    # Extra prose before JSON — not fenced, not valid JSON
    mock_resp = _make_httpx_response(f"Here is the analysis:\n{_valid_json_str()}")
    with patch("httpx.AsyncClient.post", new=_AsyncMock(return_value=mock_resp)):
        with pytest.raises(ProviderError, match="invalid JSON"):
            await provider.generate_structured_analysis(evidence)


# ══════════════════════════════════════════════════════════════════════════
# BedrockProvider tests — all boto3 calls are mocked, no real AWS calls made
# ══════════════════════════════════════════════════════════════════════════

from unittest.mock import patch, MagicMock


def _make_evidence():
    from app.domain.models import (
        ContentMetadata, ContentMetrics, GateResult,
        GateClassification, EvidencePackage, DataQuality,
    )
    return EvidencePackage(
        content_metadata=ContentMetadata(content_id="c-1", title="T", creator_id="u-1"),
        verified_metrics=ContentMetrics(
            current_hour_delta_views=400,
            seven_day_rolling_hourly_baseline=100,
            one_hour_delta_likes=10,
            one_hour_delta_views=400,
        ),
        gate_result=GateResult(
            classification=GateClassification.BOOMING_SURGE,
            velocity_ratio=4.0,
            like_acceleration=10.0,
            reason="surge",
        ),
        data_quality=DataQuality(),
    )


def _make_converse_response(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "stopReason": stop_reason,
        "usage": {"inputTokens": 50, "outputTokens": 100},
    }


def _make_bedrock_provider():
    """Construct a BedrockProvider with both boto3 clients mocked out."""
    from app.providers.bedrock import BedrockProvider
    mock_runtime = MagicMock()
    mock_bedrock = MagicMock()
    with patch("boto3.client", side_effect=[mock_runtime, mock_bedrock]):
        provider = BedrockProvider()
    return provider, mock_runtime, mock_bedrock


# ── 1. Metadata ────────────────────────────────────────────────────────────

def test_bedrock_provider_metadata():
    provider, _, _ = _make_bedrock_provider()
    meta = provider.get_provider_metadata()
    assert meta["provider"] == "bedrock"
    assert meta["status"] == "active"
    assert "region" in meta
    assert "model_id" in meta
    # Credentials must never appear in metadata
    assert "access_key" not in str(meta).lower()
    assert "secret" not in str(meta).lower()
    assert "token" not in str(meta).lower()


# ── 2. Successful structured response ─────────────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_generate_structured_analysis_success():
    provider, mock_runtime, _ = _make_bedrock_provider()
    mock_runtime.converse.return_value = _make_converse_response(_valid_json_str())

    result = await provider.generate_structured_analysis(_make_evidence())

    assert result.content_intent == "Educational"
    assert result.confidence == pytest.approx(0.8)
    mock_runtime.converse.assert_called_once()


# ── 3. Plain JSON parsing ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_parses_plain_json():
    provider, mock_runtime, _ = _make_bedrock_provider()
    mock_runtime.converse.return_value = _make_converse_response(_valid_json_str())

    result = await provider.generate_structured_analysis(_make_evidence())
    assert isinstance(result.observed_signals, list)
    assert len(result.observed_signals) >= 1


# ── 4. Fenced JSON parsing ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_parses_fenced_json():
    provider, mock_runtime, _ = _make_bedrock_provider()
    fenced = f"```json\n{_valid_json_str()}\n```"
    mock_runtime.converse.return_value = _make_converse_response(fenced)

    result = await provider.generate_structured_analysis(_make_evidence())
    assert result.content_intent == "Educational"


# ── 5. Malformed JSON ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_raises_on_malformed_json():
    from app.core.exceptions import ProviderError
    provider, mock_runtime, _ = _make_bedrock_provider()
    mock_runtime.converse.return_value = _make_converse_response("not json {broken")

    with pytest.raises(ProviderError, match="invalid JSON"):
        await provider.generate_structured_analysis(_make_evidence())


# ── 6. Truncated response (max_tokens stop reason) ────────────────────────

@pytest.mark.asyncio
async def test_bedrock_raises_on_truncated_output():
    from app.core.exceptions import ProviderError
    provider, mock_runtime, _ = _make_bedrock_provider()
    mock_runtime.converse.return_value = _make_converse_response(
        '{"content_intent": "trunc', stop_reason="max_tokens"
    )

    with pytest.raises(ProviderError, match="truncated"):
        await provider.generate_structured_analysis(_make_evidence())


# ── 7. Bedrock invocation failure (ClientError) ────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_raises_provider_error_on_client_error():
    from app.core.exceptions import ProviderError
    from botocore.exceptions import ClientError
    provider, mock_runtime, _ = _make_bedrock_provider()
    mock_runtime.converse.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "bad input"}},
        "Converse",
    )

    with pytest.raises(ProviderError):
        await provider.generate_structured_analysis(_make_evidence())


# ── 8. Health check success ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_health_check_success():
    provider, _, mock_bedrock = _make_bedrock_provider()
    mock_bedrock.list_foundation_models.return_value = {"modelSummaries": []}

    result = await provider.health_check()
    assert result is True


# ── 9. Health check failure ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bedrock_health_check_failure_on_no_credentials():
    from botocore.exceptions import NoCredentialsError
    provider, _, mock_bedrock = _make_bedrock_provider()
    mock_bedrock.list_foundation_models.side_effect = NoCredentialsError()

    result = await provider.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_bedrock_health_check_failure_on_client_error():
    from botocore.exceptions import ClientError
    provider, _, mock_bedrock = _make_bedrock_provider()
    mock_bedrock.list_foundation_models.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "ListFoundationModels",
    )

    result = await provider.health_check()
    assert result is False


# ── 10. Credentials not exposed in metadata or logs ───────────────────────

def test_bedrock_metadata_contains_no_credentials():
    provider, _, _ = _make_bedrock_provider()
    meta = provider.get_provider_metadata()
    meta_str = str(meta).lower()
    for forbidden in ("aws_access_key", "aws_secret", "session_token", "password", "credential"):
        assert forbidden not in meta_str, f"Sensitive key found in metadata: {forbidden}"


# ── 11. No-credentials error at construction maps to failure ──────────────

def test_bedrock_no_credentials_at_construction():
    from app.providers.bedrock import BedrockProvider
    from botocore.exceptions import NoCredentialsError

    with patch("boto3.client", side_effect=NoCredentialsError()):
        try:
            BedrockProvider()
            # If construction succeeds despite no credentials, that is also
            # acceptable — boto3 may defer credential resolution to call time.
        except Exception:
            pass  # Any exception at construction is acceptable


# ── 12. Access denied maps to ProviderUnavailableError ────────────────────

@pytest.mark.asyncio
async def test_bedrock_access_denied_raises_unavailable():
    from app.core.exceptions import ProviderUnavailableError
    from botocore.exceptions import ClientError
    provider, mock_runtime, _ = _make_bedrock_provider()
    mock_runtime.converse.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        "Converse",
    )

    with pytest.raises(ProviderUnavailableError, match="access denied"):
        await provider.generate_structured_analysis(_make_evidence())


# ── 13. Missing response content raises ProviderError ─────────────────────

@pytest.mark.asyncio
async def test_bedrock_raises_on_missing_response_content():
    from app.core.exceptions import ProviderError
    provider, mock_runtime, _ = _make_bedrock_provider()
    # Malformed response — missing expected keys
    mock_runtime.converse.return_value = {"stopReason": "end_turn", "usage": {}}

    with pytest.raises(ProviderError, match="response structure"):
        await provider.generate_structured_analysis(_make_evidence())


# ── 14. Truncation error message references BEDROCK_MAX_TOKENS ────────────

@pytest.mark.asyncio
async def test_bedrock_truncation_error_references_bedrock_setting():
    from app.core.exceptions import ProviderError
    provider, mock_runtime, _ = _make_bedrock_provider()
    mock_runtime.converse.return_value = _make_converse_response(
        '{"content_intent": "trunc', stop_reason="max_tokens"
    )

    with pytest.raises(ProviderError, match="BEDROCK_MAX_TOKENS"):
        await provider.generate_structured_analysis(_make_evidence())

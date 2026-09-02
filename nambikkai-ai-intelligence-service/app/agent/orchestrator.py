from app.agent.decision_engine import should_invoke_llm, should_monitor
from app.audit.models import AuditRecord
from app.core.exceptions import ProviderError, ProviderUnavailableError
from app.core.logging import get_logger
from app.domain.models import (
    AnalysisResult,
    AnalysisStatus,
    AnalyticsEvent,
    GateClassification,
    ValidationResult,
)
from app.providers.base import LLMProvider
from app.services.evidence_builder import build_evidence_package
from app.services.gating import evaluate_gate
from app.validation.output_validator import validate_output
from app.validation.policy_validator import validate_policy

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    The Agent is NOT the LLM.

    It is the control layer that:
      Observe → Evaluate → Decide → Build Evidence → Invoke LLM → Validate → Return → Audit
    """

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def run(self, event: AnalyticsEvent) -> tuple[AnalysisResult, AuditRecord]:
        # ── 1. Observe ────────────────────────────────────────────────────
        logger.info("Agent observing event", extra={"event_id": event.event_id})

        # ── 2. Evaluate (deterministic gate) ─────────────────────────────
        gate_result = evaluate_gate(event.metrics)
        logger.info("Gate result", extra={"classification": gate_result.classification})

        # ── 3. Decide ─────────────────────────────────────────────────────
        llm_invoked = False

        if gate_result.classification == GateClassification.NOMINAL:
            result = AnalysisResult(
                status=AnalysisStatus.SKIPPED_NOMINAL,
                gate_result=gate_result,
                message="Event is nominal. LLM not invoked.",
            )
            return result, self._make_audit(event, result, llm_invoked)

        if should_monitor(gate_result):
            result = AnalysisResult(
                status=AnalysisStatus.MONITORING,
                gate_result=gate_result,
                message="Event is elevated. Monitoring — LLM not invoked.",
            )
            return result, self._make_audit(event, result, llm_invoked)

        if not should_invoke_llm(gate_result):
            result = AnalysisResult(
                status=AnalysisStatus.SKIPPED_NOMINAL,
                gate_result=gate_result,
                message="Gate did not qualify for LLM invocation.",
            )
            return result, self._make_audit(event, result, llm_invoked)

        # ── 4. Build Evidence ─────────────────────────────────────────────
        evidence = build_evidence_package(
            metadata=event.metadata,
            metrics=event.metrics,
            gate_result=gate_result,
        )

        # ── 5. Invoke LLM Provider ────────────────────────────────────────
        llm_invoked = True
        try:
            editorial = await self._provider.generate_structured_analysis(evidence)
        except ProviderUnavailableError as exc:
            logger.error("Provider unavailable: %s", exc)
            result = AnalysisResult(
                status=AnalysisStatus.PROVIDER_ERROR,
                gate_result=gate_result,
                message=f"Provider unavailable: {exc}",
            )
            return result, self._make_audit(event, result, llm_invoked)
        except ProviderError as exc:
            logger.error("Provider error: %s", exc)
            result = AnalysisResult(
                status=AnalysisStatus.PROVIDER_ERROR,
                gate_result=gate_result,
                message=f"Provider error: {exc}",
            )
            return result, self._make_audit(event, result, llm_invoked)

        # ── 6. Validate ───────────────────────────────────────────────────
        output_validation = validate_output(editorial, evidence)
        policy_validation = validate_policy(editorial)

        all_failures = output_validation.failures + policy_validation.failures
        if all_failures:
            combined = ValidationResult(is_valid=False, failures=all_failures)
            result = AnalysisResult(
                status=AnalysisStatus.INVALID,
                gate_result=gate_result,
                validation_result=combined,
                message="AI output failed validation.",
            )
            return result, self._make_audit(event, result, llm_invoked)

        combined_ok = ValidationResult(is_valid=True)
        result = AnalysisResult(
            status=AnalysisStatus.SUCCESS,
            gate_result=gate_result,
            editorial_analysis=editorial,
            validation_result=combined_ok,
        )
        return result, self._make_audit(event, result, llm_invoked)

    def _make_audit(
        self,
        event: AnalyticsEvent,
        result: AnalysisResult,
        llm_invoked: bool,
    ) -> AuditRecord:
        return AuditRecord(
            event_id=event.event_id,
            content_id=event.metadata.content_id,
            gate_classification=result.gate_result.classification,
            analysis_status=result.status,
            provider_used=self._provider.get_provider_metadata()["provider"] if llm_invoked else None,
            validation_passed=result.validation_result.is_valid if result.validation_result else None,
            llm_invoked=llm_invoked,
        )

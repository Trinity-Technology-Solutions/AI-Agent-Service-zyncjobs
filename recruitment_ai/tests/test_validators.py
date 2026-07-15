"""Tests for the validators package."""
import pytest
from recruitment_ai.brains.shared.brain_result import BrainResult
from recruitment_ai.brains.shared.brain_state import BrainState, RetrievedDocuments
from recruitment_ai.validators.json_validator import extract_json, validate_json, validate_json_strict, ensure_json_fields
from recruitment_ai.validators.schema_validator import validate_brain_result, validate_brain_state
from recruitment_ai.validators.safety_validator import validate_safety, redact_pii, check_pii
from recruitment_ai.validators.citation_validator import validate_citation, validate_citations, deduplicate_citations
from recruitment_ai.validators.response_validator import validate_response, ValidationReport
from recruitment_ai.validators.hallucination_validator import validate_against_context, has_hallucination_risk


class TestJsonValidator:
    def test_extract_json_from_code_block(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        result = extract_json(text)
        assert result is not None
        assert '"key": "value"' in result

    def test_extract_json_from_plain(self):
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_extract_json_none(self):
        assert extract_json("just text") is None

    def test_validate_json_valid(self):
        assert validate_json('{"a": 1}') == {"a": 1}

    def test_validate_json_invalid(self):
        assert validate_json("not json") is None

    def test_validate_json_strict_object(self):
        assert validate_json_strict('{"a": 1}', "object") == {"a": 1}
        assert validate_json_strict("[1, 2]", "object") is None

    def test_ensure_json_fields(self):
        data = {"a": 1, "b": 2}
        assert ensure_json_fields(data, ["a", "b"]) == []
        assert ensure_json_fields(data, ["a", "c"]) == ["c"]


class TestSafetyValidator:
    def test_check_pii_email(self):
        issues = check_pii("contact me at test@email.com")
        assert any(i["type"] == "email" for i in issues)

    def test_check_pii_phone(self):
        issues = check_pii("call me at +91-9876543210")
        assert any(i["type"] == "phone" for i in issues)

    def test_check_pii_aadhaar(self):
        issues = check_pii("my aadhaar is 1234 5678 9012")
        assert any(i["type"] == "aadhaar" for i in issues)

    def test_validate_safety_clean(self):
        report = validate_safety("hello world")
        assert not report.flagged

    def test_validate_safety_pii(self):
        report = validate_safety("email me at user@example.com")
        assert report.flagged
        assert any(i["category"] == "pii" for i in report.issues)

    def test_redact_pii(self):
        result = redact_pii("email: user@test.com")
        assert "[REDACTED_EMAIL]" in result
        assert "user@test.com" not in result


class TestCitationValidator:
    def test_valid_citation(self):
        errors = validate_citation({"title": "Doc", "url": "https://example.com"})
        assert errors == []

    def test_missing_required_fields(self):
        errors = validate_citation({"title": "Doc"})
        assert any("url" in e for e in errors)

    def test_invalid_url(self):
        errors = validate_citation({"title": "Doc", "url": "ftp://bad"})
        assert any("Invalid" in e for e in errors)

    def test_deduplicate(self):
        citations = [
            {"title": "A", "url": "http://a.com"},
            {"title": "A", "url": "http://a.com"},
            {"title": "B", "url": "http://b.com"},
        ]
        deduped = deduplicate_citations(citations)
        assert len(deduped) == 2

    def test_validate_citations_empty(self):
        assert validate_citations([]) == []


class TestSchemaValidator:
    def test_validate_brain_result_success(self):
        result = BrainResult(success=True, response={"reply": "hello"})
        errors = validate_brain_result(result)
        assert errors == []

    def test_validate_brain_result_negative_tokens(self):
        result = BrainResult(success=True, response={"reply": "hi"}, tokens=-1)
        errors = validate_brain_result(result)
        assert any("token" in e for e in errors)

    def test_validate_brain_result_negative_time(self):
        result = BrainResult(success=True, response={"reply": "hi"}, execution_time=-1)
        errors = validate_brain_result(result)
        assert any("execution_time" in e for e in errors)

    def test_validate_brain_state_long_query(self):
        state = BrainState()
        state.request.query = "x" * 10001
        errors = validate_brain_state(state)
        assert len(errors) == 1
        assert "exceeds" in errors[0]

    def test_validate_brain_state_ok(self):
        state = BrainState()
        errors = validate_brain_state(state)
        assert errors == []


class TestResponseValidator:
    def test_validate_response_success(self):
        state = BrainState(intent="CHAT")
        result = BrainResult(success=True, response={"reply": "hello", "sources": []})
        report = validate_response(result, state)
        assert report.passed

    def test_validate_response_missing_reply_fields(self):
        state = BrainState(intent="CHAT")
        result = BrainResult(success=True, response={"reply": "hello"})
        report = validate_response(result, state)
        assert not report.field_warnings or "sources" in str(report.field_warnings)

    def test_validate_response_hallucination_no_rag(self):
        state = BrainState(intent="CHAT")
        result = BrainResult(success=True, response={"reply": "hello"})
        report = validate_response(result, state)
        assert report.passed


class TestHallucinationValidator:
    def test_no_context_no_issues(self):
        state = BrainState()
        result = BrainResult(success=True, response={"reply": "hello"})
        issues = validate_against_context(result, state)
        assert issues == []

    def test_with_context_and_claim(self):
        state = BrainState()
        state.retrieved_documents.chunks = [{"text": "ZyncJobs is a platform", "title": "About", "url": "http://z.com"}]
        result = BrainResult(success=True, response={"reply": "ZyncJobs is a platform that helps people"})
        issues = validate_against_context(result, state)
        assert isinstance(issues, list)

    def test_has_hallucination_risk_no_context(self):
        state = BrainState()
        result = BrainResult(success=True, response={"reply": "hello"})
        assert has_hallucination_risk(result, state) is False

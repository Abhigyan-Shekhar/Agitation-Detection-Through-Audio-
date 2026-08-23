from __future__ import annotations

import json
import logging

import pytest

from qwen_person3 import (
    Person3Config,
    Person3Error,
    QWEN_JSON_RESPONSE_FORMAT,
    QwenPerson3Analyzer,
    analyze_person2_behaviours,
    QwenResponseValidationError,
    build_qwen_prompt,
    validate_qwen_response,
)


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, responses: list[str] | None = None, *, fail: Exception | list[Exception | None] | None = None) -> None:
        self.responses = responses or []
        self.failures = fail if isinstance(fail, list) else [fail]
        self.calls = 0
        self.kwargs: list[dict] = []

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        failure = self.failures.pop(0) if self.failures else None
        if failure:
            raise failure
        return _Response(self.responses.pop(0))


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = _Chat(completions)


@pytest.fixture
def record() -> dict:
    return {
        "start": 10.2,
        "end": 27.5,
        "behaviour": "Repeated questioning",
        "score": 0.91,
        "evidence": "Phrase repeated 3 times within nearby transcript segments.",
        "text": "Where is my daughter? Where is my daughter?",
        "chunk_id": "chunk-0002",
    }


def valid_response(**overrides) -> str:
    payload = {
        "behaviour": "Repetitive Questioning",
        "start": 10.2,
        "end": 27.5,
        "validated": True,
        "severity": "Moderate",
        "confidence": 0.94,
        "evidence": "The same question occurs repeatedly within the contextual window.",
        "explanation": "The repeated question is supported by timestamped transcript segments.",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_valid_qwen_response(record):
    result = validate_qwen_response(valid_response(), record)

    assert result.behaviour == "Repetitive Questioning"
    assert result.validated is True
    assert result.severity == "Moderate"
    assert result.confidence == 0.94
    assert result.person2_evidence == record["evidence"]


def test_qwen_call_uses_qwen_json_object_mode_and_strict_prompt(record):
    completions = _Completions([valid_response()])
    analyzer = QwenPerson3Analyzer(config=Person3Config(api_key="test", model="qwen/qwen3.6-27b"), client=_Client(completions))

    result = analyzer.analyze_record(record)

    assert result.validated is True
    assert completions.kwargs[0]["model"] == "qwen/qwen3.6-27b"
    assert completions.kwargs[0]["response_format"] == QWEN_JSON_RESPONSE_FORMAT
    assert "JSON object mode is enabled" in completions.kwargs[0]["messages"][0]["content"]
    assert "Return exactly one compact JSON object" in completions.kwargs[0]["messages"][1]["content"]


def test_prompt_contains_no_think_json_example_and_allowed_schema(record):
    prompt = build_qwen_prompt(record)

    assert prompt.startswith("/no_think")
    assert "Required JSON shape example" in prompt
    assert "behaviour,start,end,validated,severity,confidence,evidence,explanation" in prompt.replace(" ", "")


def test_json_inside_markdown_fence(record):
    fenced = f"```json\n{valid_response()}\n```"

    result = validate_qwen_response(fenced, record)

    assert result.behaviour == "Repetitive Questioning"
    assert result.start == record["start"]
    assert result.end == record["end"]


def test_json_with_harmless_surrounding_text(record):
    wrapped = f"Here is the JSON:\n{valid_response()}\nDone."

    result = validate_qwen_response(wrapped, record)

    assert result.validated is True


def test_analyze_person2_behaviours_strips_observed_qwen_thinking_block(record):
    raw_content = f"""<think>
Here's a thinking process:
...
</think>
{valid_response(behaviour="Complaining", evidence="The transcript contains repeated complaints about the situation.", explanation="Complaining is supported by the supplied transcript context.")}"""
    completions = _Completions([raw_content])

    results = analyze_person2_behaviours(
        [record],
        config=Person3Config(api_key="test"),
        client=_Client(completions),
    )

    assert len(results) == 1
    assert results[0].behaviour == "Complaining"
    assert results[0].start == record["start"]
    assert results[0].end == record["end"]
    assert "thinking process" not in results[0].evidence
    assert "thinking process" not in results[0].explanation


def test_malformed_json(record):
    with pytest.raises(QwenResponseValidationError, match="valid JSON"):
        validate_qwen_response("not-json", record)


def test_empty_response_has_clear_error(record):
    with pytest.raises(QwenResponseValidationError, match="no usable content"):
        validate_qwen_response(None, record)


def test_unparseable_response_is_logged_safely(record, caplog):
    caplog.set_level(logging.WARNING, logger="qwen_person3")

    with pytest.raises(QwenResponseValidationError, match="valid JSON"):
        validate_qwen_response("Bearer secret-token not-json", record)

    assert "could not parse model response preview" in caplog.text
    assert "secret-token" not in caplog.text
    assert "bearer=<redacted>" in caplog.text.lower()


def test_missing_required_field(record):
    payload = json.loads(valid_response())
    del payload["explanation"]

    with pytest.raises(QwenResponseValidationError, match="missing required"):
        validate_qwen_response(payload, record)


def test_extra_field_is_rejected(record):
    payload = json.loads(valid_response(extra="not allowed"))

    with pytest.raises(QwenResponseValidationError, match="unsupported field"):
        validate_qwen_response(payload, record)


def test_invalid_confidence(record):
    with pytest.raises(QwenResponseValidationError, match="confidence"):
        validate_qwen_response(valid_response(confidence=1.5), record)


def test_invalid_severity(record):
    with pytest.raises(QwenResponseValidationError, match="severity"):
        validate_qwen_response(valid_response(severity="Critical"), record)


def test_api_failure(record):
    completions = _Completions(fail=TimeoutError("network timeout"))
    analyzer = QwenPerson3Analyzer(config=Person3Config(api_key="test"), client=_Client(completions))

    with pytest.raises(Person3Error, match="Qwen/Groq analysis failed"):
        analyzer.analyze_record(record)


def test_json_mode_validate_failure_retries_without_response_format(record):
    completions = _Completions(
        [valid_response()],
        fail=[RuntimeError("Error code: 400 - json_validate_failed - Failed to validate JSON."), None],
    )
    analyzer = QwenPerson3Analyzer(config=Person3Config(api_key="test"), client=_Client(completions))

    result = analyzer.analyze_record(record)

    assert result.validated is True
    assert completions.calls == 2
    assert completions.kwargs[0]["response_format"] == QWEN_JSON_RESPONSE_FORMAT
    assert "response_format" not in completions.kwargs[1]


def test_timestamp_preservation(record):
    with pytest.raises(QwenResponseValidationError, match="preserve"):
        validate_qwen_response(valid_response(start=11.0), record)


def test_multiple_behaviour_records_deduplicates_duplicate_evidence(record):
    completions = _Completions([valid_response()])
    analyzer = QwenPerson3Analyzer(config=Person3Config(api_key="test"), client=_Client(completions))

    results = analyzer.analyze_batch([record, dict(record)])

    assert len(results) == 2
    assert results[0] == results[1]
    assert completions.calls == 1

from __future__ import annotations

import json

import pytest

from qwen_person3 import (
    Person3Config,
    Person3Error,
    QwenPerson3Analyzer,
    QwenResponseValidationError,
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
    def __init__(self, responses: list[str] | None = None, *, fail: Exception | None = None) -> None:
        self.responses = responses or []
        self.fail = fail
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        if self.fail:
            raise self.fail
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


def test_malformed_json(record):
    with pytest.raises(QwenResponseValidationError, match="valid JSON"):
        validate_qwen_response("not-json", record)


def test_missing_required_field(record):
    payload = json.loads(valid_response())
    del payload["explanation"]

    with pytest.raises(QwenResponseValidationError, match="missing required"):
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

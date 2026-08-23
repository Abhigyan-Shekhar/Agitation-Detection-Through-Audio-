"""Person 3 Qwen/Groq validation for Person 2 behaviour evidence.

This module consumes only the stable ``Person2AnalysisResult.behaviour_contract``
payload and returns validated, timestamp-preserving behaviour decisions. It does
not run transcription, chunking, embedding, or initial behaviour detection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import re
from typing import Any, Iterable


DEFAULT_QWEN_MODEL = "qwen/qwen3-32b"
VALID_SEVERITIES = {"Insufficient", "Low", "Mild", "Moderate", "High", "Severe"}
REQUIRED_RESPONSE_FIELDS = {
    "behaviour",
    "start",
    "end",
    "validated",
    "severity",
    "confidence",
    "evidence",
    "explanation",
}


class Person3Error(RuntimeError):
    """Base class for Person 3 failures that dashboards can display."""


class MissingGroqApiKeyError(Person3Error):
    """Raised when GROQ_API_KEY is required but not configured."""


class QwenResponseValidationError(Person3Error):
    """Raised when Qwen returns malformed or unsafe structured output."""


@dataclass(frozen=True)
class Person3Config:
    """Runtime configuration for Qwen-on-Groq final analysis."""

    api_key: str | None = None
    model: str = DEFAULT_QWEN_MODEL
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "Person3Config":
        """Load Groq/Qwen settings from environment variables."""
        return cls(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
            timeout_seconds=float(os.getenv("QWEN_TIMEOUT_SECONDS", "30")),
        )


@dataclass(frozen=True)
class FinalBehaviourResult:
    """Validated final behaviour decision shown by the MVP dashboard."""

    behaviour: str
    start: float
    end: float
    validated: bool
    severity: str
    confidence: float
    evidence: str
    explanation: str
    initial_behaviour: str
    initial_score: float | None
    person2_evidence: str
    transcript: str
    chunk_id: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON/table-ready representation."""
        return asdict(self)


class QwenPerson3Analyzer:
    """Analyze Person 2 behaviour evidence with Qwen through Groq."""

    def __init__(self, config: Person3Config | None = None, *, client: Any | None = None) -> None:
        self.config = config or Person3Config.from_env()
        self._client = client
        self._cache: dict[str, FinalBehaviourResult] = {}

    @property
    def client(self) -> Any:
        """Lazily build the official Groq client."""
        if self._client is not None:
            return self._client
        if not self.config.api_key:
            raise MissingGroqApiKeyError("GROQ_API_KEY is not configured; set it before running Qwen analysis.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise Person3Error("The groq package is not installed. Install requirements.txt first.") from exc
        self._client = Groq(api_key=self.config.api_key, timeout=self.config.timeout_seconds)
        return self._client

    def analyze_batch(self, behaviour_records: Iterable[dict[str, Any]]) -> list[FinalBehaviourResult]:
        """Analyze multiple Person 2 records, reusing results for duplicate evidence."""
        results: list[FinalBehaviourResult] = []
        for record in behaviour_records:
            cache_key = _record_cache_key(record)
            if cache_key not in self._cache:
                self._cache[cache_key] = self.analyze_record(record)
            results.append(self._cache[cache_key])
        return results

    def analyze_record(self, record: dict[str, Any]) -> FinalBehaviourResult:
        """Analyze one Person 2 behaviour evidence record."""
        prompt = build_qwen_prompt(record)
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return validate_qwen_response(content, record)
        except Person3Error:
            raise
        except Exception as exc:  # noqa: BLE001 - surface API/network failures to dashboard
            raise Person3Error(f"Qwen/Groq analysis failed: {exc}") from exc


def analyze_person2_behaviours(
    behaviour_records: Iterable[dict[str, Any]],
    *,
    config: Person3Config | None = None,
    client: Any | None = None,
) -> list[FinalBehaviourResult]:
    """Convenience integration point for Person 2's ``behaviour_contract()`` output."""
    return QwenPerson3Analyzer(config=config, client=client).analyze_batch(behaviour_records)


def build_qwen_prompt(record: dict[str, Any]) -> str:
    """Build the user prompt for one Person 2 behaviour record."""
    payload = {
        "start": record.get("start"),
        "end": record.get("end"),
        "initial_behaviour": record.get("behaviour"),
        "initial_score": record.get("score"),
        "score_type": record.get("score_type"),
        "person2_evidence": record.get("evidence"),
        "transcript_context": record.get("text"),
        "chunk_id": record.get("chunk_id"),
        "repetition": record.get("repetition"),
    }
    return (
        "Evaluate this Person 2 audio-behaviour evidence as decision support, not medical diagnosis. "
        "Use only supplied evidence/transcript. Do not invent behaviours. If evidence is insufficient, set "
        "validated=false, severity='Insufficient', and explain what is missing. Preserve the exact start/end timestamps. "
        "Return only JSON with keys: behaviour, start, end, validated, severity, confidence, evidence, explanation.\n\n"
        f"Person 2 evidence JSON:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def validate_qwen_response(raw_content: str | dict[str, Any], source_record: dict[str, Any]) -> FinalBehaviourResult:
    """Validate and normalize Qwen JSON without silently accepting malformed output."""
    data = _parse_json_object(raw_content)
    missing = REQUIRED_RESPONSE_FIELDS.difference(data)
    if missing:
        raise QwenResponseValidationError(f"Qwen response is missing required field(s): {', '.join(sorted(missing))}")

    start = float(source_record["start"])
    end = float(source_record["end"])
    output_start = float(data["start"])
    output_end = float(data["end"])
    if abs(output_start - start) > 1e-6 or abs(output_end - end) > 1e-6:
        raise QwenResponseValidationError("Qwen response did not preserve the supplied timestamps.")

    confidence = float(data["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise QwenResponseValidationError("Qwen confidence must be between 0.0 and 1.0.")

    severity = str(data["severity"]).strip().title()
    if severity not in VALID_SEVERITIES:
        raise QwenResponseValidationError(f"Qwen severity must be one of: {', '.join(sorted(VALID_SEVERITIES))}.")

    return FinalBehaviourResult(
        behaviour=str(data["behaviour"]).strip() or str(source_record.get("behaviour", "Unknown behaviour")),
        start=start,
        end=end,
        validated=bool(data["validated"]),
        severity=severity,
        confidence=round(confidence, 4),
        evidence=str(data["evidence"]).strip(),
        explanation=str(data["explanation"]).strip(),
        initial_behaviour=str(source_record.get("behaviour", "")),
        initial_score=_optional_float(source_record.get("score")),
        person2_evidence=str(source_record.get("evidence", "")),
        transcript=str(source_record.get("text", "")),
        chunk_id=str(source_record.get("chunk_id")) if source_record.get("chunk_id") is not None else None,
    )


def _parse_json_object(raw_content: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_content, dict):
        return raw_content
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)
        if not match:
            raise QwenResponseValidationError("Qwen response was not valid JSON.") from None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise QwenResponseValidationError("Qwen response was not valid JSON.") from exc


def _system_prompt() -> str:
    return (
        "You are Person 3 in a research audio-analysis pipeline. Validate Person 2 behaviour evidence conservatively. "
        "This is decision support, not diagnosis. Return strict JSON only."
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _record_cache_key(record: dict[str, Any]) -> str:
    stable = json.dumps(
        {
            "start": record.get("start"),
            "end": record.get("end"),
            "behaviour": record.get("behaviour"),
            "evidence": record.get("evidence"),
            "text": record.get("text"),
            "score": record.get("score"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()

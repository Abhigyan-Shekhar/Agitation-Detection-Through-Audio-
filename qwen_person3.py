"""Person 3 Qwen/Groq validation for Person 2 behaviour evidence.

This module consumes only the stable ``Person2AnalysisResult.behaviour_contract``
payload and returns validated, timestamp-preserving behaviour decisions. It does
not run transcription, chunking, embedding, or initial behaviour detection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import re
from typing import Any, Iterable


DEFAULT_QWEN_MODEL = "qwen/qwen3.6-27b"
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
QWEN_JSON_RESPONSE_FORMAT = {"type": "json_object"}
RAW_RESPONSE_LOG_CHARS = 2000
_LOGGER = logging.getLogger(__name__)


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
    reasoning_format: str | None = "hidden"

    @classmethod
    def from_env(cls) -> "Person3Config":
        """Load Groq/Qwen settings from environment variables."""
        return cls(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL),
            timeout_seconds=float(os.getenv("QWEN_TIMEOUT_SECONDS", "30")),
            reasoning_format=os.getenv("QWEN_REASONING_FORMAT", "hidden") or None,
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
        if importlib.util.find_spec("groq") is None:
            raise Person3Error("The groq package is not installed. Install requirements.txt first.")
        groq_module = importlib.import_module("groq")
        self._client = groq_module.Groq(api_key=self.config.api_key, timeout=self.config.timeout_seconds)
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
            response = self._create_completion(prompt, use_json_mode=True)
            content = _extract_message_content(response)
            _log_raw_model_response(content)
            return validate_qwen_response(content, record)
        except Person3Error:
            raise
        except Exception as exc:  # noqa: BLE001 - surface API/network failures to dashboard
            if _looks_like_groq_json_mode_failure(exc):
                try:
                    response = self._create_completion(prompt, use_json_mode=False)
                    content = _extract_message_content(response)
                    _log_raw_model_response(content, retry=True)
                    return validate_qwen_response(content, record)
                except Person3Error:
                    raise
                except Exception as retry_exc:  # noqa: BLE001
                    raise Person3Error(f"Qwen/Groq analysis failed after JSON-mode retry: {retry_exc}") from retry_exc
            raise Person3Error(f"Qwen/Groq analysis failed: {exc}") from exc

    def _create_completion(self, prompt: str, *, use_json_mode: bool) -> Any:
        """Call Groq with Qwen-compatible JSON mode or a plain-JSON fallback."""
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _system_prompt(use_json_mode=use_json_mode)},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 700,
        }
        if self.config.reasoning_format:
            kwargs["reasoning_format"] = self.config.reasoning_format
        if use_json_mode:
            kwargs["response_format"] = QWEN_JSON_RESPONSE_FORMAT
        return self.client.chat.completions.create(**kwargs)


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
    example = {
        "behaviour": payload["initial_behaviour"] or "Unknown behaviour",
        "start": payload["start"],
        "end": payload["end"],
        "validated": True,
        "severity": "Moderate",
        "confidence": 0.94,
        "evidence": "Short evidence string grounded only in the supplied transcript/evidence.",
        "explanation": "Short explanation of why the evidence supports or does not support the behaviour.",
    }
    return (
        "/no_think\n"
        "Evaluate this Person 2 audio-behaviour evidence as research decision support, not medical diagnosis. "
        "Use only supplied evidence/transcript. Do not invent behaviours. If evidence is insufficient, use "
        "validated=false, severity=\"Insufficient\", confidence between 0 and 1, and explain what is missing. "
        "Preserve the exact start/end timestamp numbers from the input.\n\n"
        "Return exactly one compact JSON object. The first character must be `{` and the last character must be `}`. "
        "Do not output <think> blocks, hidden reasoning text, markdown, code fences, comments, XML/thinking tags, "
        "or any text before or after the JSON object. The JSON object must contain exactly these keys: "
        "behaviour, start, end, validated, severity, confidence, evidence, explanation. "
        f"Allowed severity values: {', '.join(sorted(VALID_SEVERITIES))}.\n\n"
        f"Required JSON shape example:\n{json.dumps(example, separators=(',', ':'))}\n\n"
        f"Person 2 evidence JSON:\n{json.dumps(payload, indent=2, sort_keys=True)}"
    )


def validate_qwen_response(raw_content: str | dict[str, Any] | None, source_record: dict[str, Any]) -> FinalBehaviourResult:
    """Validate and normalize Qwen JSON without silently accepting malformed output."""
    try:
        data = _parse_json_object(raw_content)
    except QwenResponseValidationError:
        _log_raw_model_response(raw_content, parse_error=True)
        raise
    missing = REQUIRED_RESPONSE_FIELDS.difference(data)
    if missing:
        raise QwenResponseValidationError(f"Qwen response is missing required field(s): {', '.join(sorted(missing))}")
    extras = set(data).difference(REQUIRED_RESPONSE_FIELDS)
    if extras:
        raise QwenResponseValidationError(f"Qwen response included unsupported field(s): {', '.join(sorted(extras))}")

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
        validated=_coerce_bool(data["validated"]),
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


def _parse_json_object(raw_content: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw_content, dict):
        return raw_content
    if raw_content is None or not str(raw_content).strip():
        raise QwenResponseValidationError("Qwen returned no usable content to parse as JSON.")

    text = _remove_think_blocks(str(raw_content).strip()).strip()
    if not text:
        raise QwenResponseValidationError("Qwen returned only reasoning/thinking text and no usable JSON content.")
    candidate = _strip_markdown_json_fence(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        extracted = _extract_first_balanced_json_object(candidate)
        if extracted is None:
            raise QwenResponseValidationError("Qwen response was not valid JSON.") from None
        try:
            parsed = json.loads(extracted)
        except json.JSONDecodeError as exc:
            raise QwenResponseValidationError("Qwen response was not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise QwenResponseValidationError("Qwen response JSON must be an object with the Person 3 schema.")
    return parsed


def _remove_think_blocks(text: str) -> str:
    """Remove Qwen reasoning blocks without using them as dashboard content."""
    return re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _strip_markdown_json_fence(text: str) -> str:
    fence_match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _extract_first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_message_content(response: Any) -> str | None:
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise QwenResponseValidationError("Groq response did not contain message content for Qwen analysis.") from exc


def _log_raw_model_response(raw_content: Any, *, retry: bool = False, parse_error: bool = False) -> None:
    label = "retry raw" if retry else "raw"
    if parse_error:
        label = "unparseable raw"
    preview = _safe_response_preview(raw_content)
    _LOGGER.debug("Qwen Person 3 %s response preview: %s", label, preview)
    if parse_error:
        _LOGGER.warning("Qwen Person 3 could not parse model response preview: %s", preview)


def _safe_response_preview(raw_content: Any) -> str:
    if raw_content is None:
        return "<empty>"
    text = str(raw_content)
    text = re.sub(r"(?i)(api[_-]?key|authorization|bearer)\s*(?:[:=]\s*)?[^\s,}]+", r"\1=<redacted>", text)
    if len(text) > RAW_RESPONSE_LOG_CHARS:
        return text[:RAW_RESPONSE_LOG_CHARS] + "...<truncated>"
    return text


def _system_prompt(*, use_json_mode: bool = True) -> str:
    mode_note = (
        "Groq JSON object mode is enabled; your entire response must be one valid JSON object."
        if use_json_mode
        else "JSON object mode retry is disabled; still return one valid JSON object only."
    )
    return (
        "/no_think\n"
        "You are Person 3 in a research audio-analysis pipeline. Validate Person 2 behaviour evidence conservatively. "
        "This is decision support, not diagnosis. "
        f"{mode_note} Do not include <think> blocks, markdown, prose, code fences, or hidden reasoning. "
        "The first output character must be `{` and the last output character must be `}`. "
        "Output JSON only with keys behaviour, start, end, validated, severity, confidence, evidence, explanation."
    )


def _looks_like_groq_json_mode_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return "json_validate_failed" in message or "failed to validate json" in message


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise QwenResponseValidationError("Qwen validated field must be a boolean.")


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

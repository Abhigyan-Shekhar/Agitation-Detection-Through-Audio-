"""Person 3 Qwen/Groq validation for Person 2 behaviour evidence.

This module consumes only the stable ``Person2AnalysisResult.behaviour_contract``
payload and returns validated behaviour decisions whose timestamps are computed
from selected source transcript units. It does not run transcription, chunking,
embedding, or initial behaviour detection.
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
from typing import Any, Callable, Iterable

import config


DEFAULT_QWEN_MODEL = "qwen/qwen3.6-27b"
VALID_SEVERITIES = {"Insufficient", "Low", "Mild", "Moderate", "High", "Severe"}
REQUIRED_RESPONSE_FIELDS = {
    "behaviour",
    "support",
    "evidence_segment_ids",
    "severity",
    "confidence",
    "evidence",
    "explanation",
}
VALID_SUPPORT = {"supported", "unsupported", "insufficient"}
QWEN_JSON_RESPONSE_FORMAT = {"type": "json_object"}
QWEN_MAX_COMPLETION_TOKENS = 1024
QWEN_REASONING_EFFORT = "none"
QWEN_REASONING_FORMAT = "hidden"
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
    evidence_segment_ids: list[str] | None = None
    support: str = "supported"
    model_support_score: float | None = None
    calibrated_confidence: float | None = None
    source_segment_ids: list[str] | None = None
    evidence_segments: list[dict[str, Any]] | None = None
    speaker_id: int | str | None = None
    speaker_label: str | None = None
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

    def analyze_batch(
        self,
        behaviour_records: Iterable[dict[str, Any]],
        *,
        progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> list[FinalBehaviourResult]:
        """Analyze multiple Person 2 records, reusing results for duplicate evidence."""
        records = list(behaviour_records)
        results: list[FinalBehaviourResult] = []
        total = len(records)
        for index, record in enumerate(records):
            if progress_callback is not None:
                progress_callback(index, total, record)
            cache_key = _record_cache_key(record)
            if cache_key not in self._cache:
                self._cache[cache_key] = self.analyze_record(record)
            results.append(self._cache[cache_key])
            if progress_callback is not None:
                progress_callback(index + 1, total, record)
        return deduplicate_final_results(results)

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
            "max_completion_tokens": QWEN_MAX_COMPLETION_TOKENS,
        }
        if _supports_qwen_reasoning_controls(self.config.model):
            kwargs["reasoning_effort"] = QWEN_REASONING_EFFORT
            kwargs["reasoning_format"] = QWEN_REASONING_FORMAT
        if use_json_mode:
            kwargs["response_format"] = QWEN_JSON_RESPONSE_FORMAT
        return self.client.chat.completions.create(**kwargs)


def analyze_person2_behaviours(
    behaviour_records: Iterable[dict[str, Any]],
    *,
    config: Person3Config | None = None,
    client: Any | None = None,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[FinalBehaviourResult]:
    """Convenience integration point for Person 2's ``behaviour_contract()`` output."""
    return QwenPerson3Analyzer(config=config, client=client).analyze_batch(
        behaviour_records,
        progress_callback=progress_callback,
    )


def build_qwen_prompt(record: dict[str, Any]) -> str:
    """Build the user prompt for one Person 2 behaviour record."""
    source_units = _source_units(record)
    payload = {
        "initial_behaviour": record.get("behaviour"),
        "initial_score": record.get("score"),
        "score_type": record.get("score_type"),
        "person2_evidence": record.get("evidence"),
        "evidence_source_segment_ids": record.get("source_segment_ids"),
        "context_start": record.get("context_start"),
        "context_end": record.get("context_end"),
        "transcript_context": record.get("text"),
        "source_transcript_units": source_units,
        "chunk_id": record.get("chunk_id"),
        "repetition": record.get("repetition"),
        "acoustic_evidence": record.get("acoustic"),
        "speaker_id": record.get("speaker_id"),
        "speaker_label": record.get("speaker_label"),
    }
    example = {
        "behaviour": payload["initial_behaviour"] or "Unknown behaviour",
        "support": "supported",
        "evidence_segment_ids": [source_units[0]["id"]] if source_units else [],
        "severity": "Moderate",
        "confidence": 0.94,
        "evidence": "Short evidence string grounded only in the supplied transcript/evidence.",
        "explanation": "Short explanation of why the evidence supports or does not support the behaviour.",
    }
    return (
        "/no_think\n"
        "Evaluate this Person 2 audio-behaviour evidence as research decision support, not medical diagnosis. "
        "Use only supplied evidence/transcript. Acoustic features are measured from the source recording: use them only when explicitly supplied, and do not infer unreported vocal tone from transcript wording. Do not invent behaviours. If evidence is insufficient, use "
        "support=\"insufficient\", severity=\"Insufficient\", confidence between 0 and 1 as an uncalibrated model support score, and explain what is missing. "
        "Do not invent, modify, or return timestamps. Select only source_transcript_units IDs that directly support the decision.\n\n"
        "Return exactly one compact JSON object. The first character must be `{` and the last character must be `}`. "
        "Do not return markdown, code fences, comments, XML/thinking tags, "
        "or any text before or after the JSON object. The JSON object must contain exactly these keys: "
        "behaviour, support, evidence_segment_ids, severity, confidence, evidence, explanation. The confidence field is an uncalibrated verifier support score, not a clinical probability. "
        f"Allowed support values: {', '.join(sorted(VALID_SUPPORT))}. "
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

    support = str(data["support"]).strip().lower()
    if support not in VALID_SUPPORT:
        raise QwenResponseValidationError(f"Qwen support must be one of: {', '.join(sorted(VALID_SUPPORT))}.")
    source_units = _source_units(source_record)
    units_by_id = {str(unit["id"]): unit for unit in source_units}
    selected_ids = _coerce_segment_ids(data["evidence_segment_ids"])
    unknown_ids = [segment_id for segment_id in selected_ids if segment_id not in units_by_id]
    if unknown_ids:
        raise QwenResponseValidationError(f"Qwen selected unknown evidence segment id(s): {', '.join(unknown_ids)}")
    if support == "supported":
        if not selected_ids:
            raise QwenResponseValidationError("Qwen must select evidence_segment_ids when support is supported.")
        selected_units = [units_by_id[segment_id] for segment_id in selected_ids]
    else:
        selected_units = source_units
    start = min(float(unit["start"]) for unit in selected_units) if selected_units else float(source_record["start"])
    end = max(float(unit["end"]) for unit in selected_units) if selected_units else float(source_record["end"])

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
        validated=support == "supported",
        severity=severity,
        confidence=round(confidence, 4),
        support=support,
        model_support_score=round(confidence, 4),
        calibrated_confidence=None,
        evidence=str(data["evidence"]).strip(),
        explanation=str(data["explanation"]).strip(),
        initial_behaviour=str(source_record.get("behaviour", "")),
        initial_score=_optional_float(source_record.get("score")),
        person2_evidence=str(source_record.get("evidence", "")),
        transcript=str(source_record.get("text", "")),
        chunk_id=str(source_record.get("chunk_id")) if source_record.get("chunk_id") is not None else None,
        evidence_segment_ids=selected_ids,
        source_segment_ids=list(source_record.get("source_segment_ids") or selected_ids),
        evidence_segments=source_record.get("evidence_segments"),
        speaker_id=source_record.get("speaker_id"),
        speaker_label=source_record.get("speaker_label"),
    )


def _source_units(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_units = record.get("evidence_segments")
    if isinstance(raw_units, list) and raw_units:
        units = []
        for index, unit in enumerate(raw_units):
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("id") or f"seg-{index:06d}")
            units.append({
                "id": unit_id,
                "start": float(unit["start"]),
                "end": float(unit["end"]),
                "text": str(unit.get("text", "")),
                "speaker_id": unit.get("speaker_id"),
                "speaker_label": unit.get("speaker_label"),
            })
        if units:
            return units
    source_ids = record.get("source_segment_ids") or []
    if isinstance(source_ids, str):
        source_ids = [source_ids]
    return [{
        "id": str(source_ids[0]) if source_ids else "seg-unknown",
        "start": float(record["start"]),
        "end": float(record["end"]),
        "text": str(record.get("text", "")),
    }]


def _coerce_segment_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise QwenResponseValidationError("Qwen evidence_segment_ids must be a list.")
    return [str(item) for item in value]


def deduplicate_final_results(results: list[FinalBehaviourResult], *, iou_threshold: float | None = None) -> list[FinalBehaviourResult]:
    """Remove exact and near duplicate final rows after model validation."""
    threshold = config.PERSON2_DEDUPE_IOU_THRESHOLD if iou_threshold is None else iou_threshold
    exact: dict[tuple[str, tuple[str, ...] | tuple[float, float]], FinalBehaviourResult] = {}
    for result in results:
        ids = tuple(sorted(result.evidence_segment_ids or []))
        key = (result.behaviour, ids or (round(result.start, 1), round(result.end, 1)))
        exact[key] = _merge_final_pair(exact[key], result) if key in exact else result
    merged: list[FinalBehaviourResult] = []
    for result in sorted(exact.values(), key=lambda item: (item.start, item.end, item.behaviour, -float(item.model_support_score if item.model_support_score is not None else item.confidence))):
        match_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if existing.behaviour == result.behaviour
                and _interval_iou(existing.start, existing.end, result.start, result.end) >= threshold
            ),
            None,
        )
        if match_index is None:
            merged.append(result)
        else:
            merged[match_index] = _merge_final_pair(merged[match_index], result)
    return sorted(merged, key=lambda item: (item.start, item.end, item.behaviour))


def _merge_final_pair(left: FinalBehaviourResult, right: FinalBehaviourResult) -> FinalBehaviourResult:
    strongest = _select_primary_final(left, right)
    other = right if strongest is left else left
    ids = list(dict.fromkeys([*(strongest.evidence_segment_ids or []), *(other.evidence_segment_ids or [])]))
    evidence = strongest.evidence
    if other.evidence and other.evidence != evidence:
        evidence = f"{evidence} Additional duplicate evidence: {other.evidence}"
    return FinalBehaviourResult(
        behaviour=strongest.behaviour,
        start=min(left.start, right.start),
        end=max(left.end, right.end),
        validated=strongest.support == "supported",
        severity=strongest.severity,
        confidence=float(strongest.model_support_score if strongest.model_support_score is not None else strongest.confidence),
        support=strongest.support,
        model_support_score=strongest.model_support_score if strongest.model_support_score is not None else strongest.confidence,
        calibrated_confidence=strongest.calibrated_confidence,
        evidence=evidence,
        explanation=strongest.explanation,
        initial_behaviour=strongest.initial_behaviour,
        initial_score=strongest.initial_score,
        person2_evidence=strongest.person2_evidence,
        transcript=strongest.transcript if len(strongest.transcript) >= len(other.transcript) else other.transcript,
        chunk_id=strongest.chunk_id,
        evidence_segment_ids=ids,
        source_segment_ids=list(dict.fromkeys([*(strongest.source_segment_ids or []), *(other.source_segment_ids or [])])),
        evidence_segments=_merge_evidence_segments(strongest.evidence_segments, other.evidence_segments),
        speaker_id=strongest.speaker_id if strongest.speaker_id is not None else other.speaker_id,
        speaker_label=strongest.speaker_label if strongest.speaker_label is not None else other.speaker_label,
        error=strongest.error or other.error,
    )


def _select_primary_final(left: FinalBehaviourResult, right: FinalBehaviourResult) -> FinalBehaviourResult:
    precedence = {"supported": 2, "insufficient": 1, "unsupported": 0}
    l_sup = left.support if left.support in precedence else ("supported" if left.validated else "unsupported")
    r_sup = right.support if right.support in precedence else ("supported" if right.validated else "unsupported")
    if precedence[l_sup] != precedence[r_sup]:
        return left if precedence[l_sup] > precedence[r_sup] else right
    l_score = float(left.model_support_score if left.model_support_score is not None else left.confidence)
    r_score = float(right.model_support_score if right.model_support_score is not None else right.confidence)
    return left if l_score >= r_score else right


def _merge_evidence_segments(left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    merged: dict[str, dict[str, Any]] = {}
    for item in [*(left or []), *(right or [])]:
        if isinstance(item, dict):
            merged[str(item.get("id", len(merged)))] = item
    return list(merged.values()) or None


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def _parse_json_object(raw_content: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw_content, dict):
        return raw_content
    if raw_content is None or not str(raw_content).strip():
        raise QwenResponseValidationError("Qwen returned no usable content to parse as JSON.")

    text = _remove_thinking_sections(str(raw_content)).strip()
    if not text:
        raise QwenResponseValidationError("Qwen returned no usable content to parse as JSON after removing thinking sections.")
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


def _remove_thinking_sections(text: str) -> str:
    """Remove Qwen reasoning blocks before parsing or displaying model output."""
    without_closed_blocks = re.sub(r"(?is)<think\b[^>]*>.*?</think>", "", text)
    return re.sub(r"(?is)<think\b[^>]*>.*$", "", without_closed_blocks)


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
    text = _remove_thinking_sections(str(raw_content)).strip() or "<thinking redacted>"
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
        f"{mode_note} Do not include markdown, prose, code fences, or hidden reasoning. "
        "The first output character must be `{` and the last output character must be `}`. "
        "Output JSON only with keys behaviour, support, evidence_segment_ids, severity, confidence, evidence, explanation."
    )


def _supports_qwen_reasoning_controls(model: str) -> bool:
    return model.strip().lower() == DEFAULT_QWEN_MODEL


def _looks_like_groq_json_mode_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return "json_validate_failed" in message or "failed to validate json" in message


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
            "source_segment_ids": record.get("source_segment_ids"),
            "evidence_segments": record.get("evidence_segments"),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()

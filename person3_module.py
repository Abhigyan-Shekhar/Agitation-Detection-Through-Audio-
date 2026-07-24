"""Person 3: Gemini linguistic analysis (optional ablation module).

This module is DISABLED from the primary execution path.
To enable for ablation experiments, set:
    ENABLE_GEMINI_COMPARISON=true

When enabled, the dashboard runs both the deterministic rule-based
pipeline and Gemini in parallel, allowing side-by-side comparison.
Never set this true in production — it adds latency and API cost.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol


class GeminiClientProtocol(Protocol):
    @property
    def models(self) -> Any: ...


@dataclass(frozen=True)
class GeminiAnalysis:
    emotion: str
    agitation_score: float
    behaviours: list[str]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded_score(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number from 0 to 1")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number from 0 to 1") from exc
    if not 0 <= result <= 1:
        raise ValueError(f"{name} must be from 0 to 1")
    return result


def _finite_nonnegative(value: Any) -> float | None:
    """Return a usable measurement, or None when that measurement is unavailable."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _distress_event_count(features: Mapping[str, Any]) -> int | None:
    """Read explicit distress-event fields without inferring events from audio."""
    count = _finite_nonnegative(features.get("distress_event_count"))
    if count is not None:
        return int(count)

    events = features.get("distress_events")
    if isinstance(events, (list, tuple, set)):
        return len(events)

    # Optional generic event lists can be supplied by a later upstream detector.
    events = features.get("events")
    if not isinstance(events, (list, tuple, set)):
        return None
    distress_terms = ("scream", "shout", "cry", "sob", "distress", "groan")
    return sum(
        any(term in str(event).lower() for term in distress_terms) for event in events
    )


def compute_acoustic_score(acoustic_features: Mapping[str, Any] | None) -> float:
    """Create a transparent 0--1 fallback score from available acoustic signals.

    The existing pipeline provides RMS energy and pitch variance. Future upstream
    stages may additionally provide ``speech_rate``/``speech_rate_wpm`` and
    distress-event fields. Only measurements that are actually present contribute
    to the weighted average, so missing optional signals do not imply calmness.
    """
    features = acoustic_features or {}
    weighted_signals: list[tuple[float, float]] = []

    rms_energy = _finite_nonnegative(features.get("rms_energy"))
    if rms_energy is not None:
        # Normalized pipeline audio: RMS >= 0.30 is treated as high energy.
        weighted_signals.append((_clamp_unit(rms_energy / 0.30), 0.35))

    pitch_variance = _finite_nonnegative(features.get("pitch_variance"))
    if pitch_variance is not None:
        # 2,500 Hz^2 corresponds roughly to a 50 Hz pitch standard deviation.
        weighted_signals.append((_clamp_unit(pitch_variance / 2500.0), 0.30))

    speech_rate = _finite_nonnegative(
        features.get("speech_rate_wpm", features.get("speech_rate"))
    )
    if speech_rate is not None:
        # Rates at or below 100 WPM do not raise this agitation heuristic; 220+ do.
        weighted_signals.append((_clamp_unit((speech_rate - 100.0) / 120.0), 0.20))

    distress_count = _distress_event_count(features)
    if distress_count is not None:
        weighted_signals.append((_clamp_unit(distress_count / 3.0), 0.15))

    if not weighted_signals:
        return 0.0
    weighted_total = sum(score * weight for score, weight in weighted_signals)
    total_weight = sum(weight for _, weight in weighted_signals)
    return round(weighted_total / total_weight, 4)


def resolve_acoustic_score(
    acoustic_score: float | None, acoustic_features: Mapping[str, Any] | None
) -> float:
    """Prefer an upstream score, otherwise calculate the Person 3 fallback."""
    if acoustic_score is not None:
        return _bounded_score(acoustic_score, "acoustic_score")
    return compute_acoustic_score(acoustic_features)


def validate_gemini_response(payload: Mapping[str, Any]) -> GeminiAnalysis:
    """Validate the required Gemini JSON contract."""
    required = {"emotion", "agitation_score", "behaviours", "reasoning"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Gemini response missing: {sorted(missing)}")
    if not isinstance(payload["emotion"], str) or not payload["emotion"].strip():
        raise ValueError("emotion must be a non-empty string")
    if not isinstance(payload["reasoning"], str):
        raise ValueError("reasoning must be a string")
    if not isinstance(payload["behaviours"], list) or not all(isinstance(x, str) for x in payload["behaviours"]):
        raise ValueError("behaviours must be a list of strings")
    return GeminiAnalysis(
        emotion=payload["emotion"].strip(),
        agitation_score=_bounded_score(payload["agitation_score"], "agitation_score"),
        behaviours=[item.strip() for item in payload["behaviours"] if item.strip()],
        reasoning=payload["reasoning"].strip(),
    )


def _parse_json(text: str) -> Mapping[str, Any]:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Gemini did not return valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Gemini JSON must be an object")
    return payload


class GeminiBehaviourAnalyzer:
    """Gemini adapter. The client can be injected for offline tests."""
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash", client: GeminiClientProtocol | None = None) -> None:
        self.model = model
        if client is not None:
            self.client = client
            return
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY before using Gemini analysis")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError("Install google-genai to use Gemini analysis") from exc
        self.client = genai.Client(api_key=key)
        self._config = types.GenerateContentConfig(response_mime_type="application/json")

    def analyze(self, transcript: str | None, acoustic_features: Mapping[str, Any] | None, acoustic_score: float | None = None) -> dict[str, Any]:
        if acoustic_score is not None:
            _bounded_score(acoustic_score, "acoustic_score")
        prompt = (
            "You are a cautious linguistic agitation analyst. This is decision support, not a diagnosis. "
            "Return exactly one JSON object with emotion (short string), agitation_score (0 to 1), "
            "behaviours (array of neutral observable labels), and reasoning (brief evidence-based explanation). "
            "Do not infer facts absent from the input.\n"
            f"Transcript: {transcript or ''}\nAcoustic features: {json.dumps(dict(acoustic_features or {}), default=str)}\n"
            f"Optional acoustic score: {acoustic_score}"
        )
        kwargs: dict[str, Any] = {"model": self.model, "contents": prompt}
        if hasattr(self, "_config"):
            kwargs["config"] = self._config
        response = self.client.models.generate_content(**kwargs)
        return validate_gemini_response(_parse_json(response.text)).to_dict()


CMAI_BEHAVIOUR_MAP = {
    "pacing": "Physically non-aggressive: pacing/aimless movement",
    "restlessness": "Physically non-aggressive: general restlessness",
    "repetitive mannerism": "Physically non-aggressive: repetitive mannerism",
    "repetitive question": "Verbally non-aggressive: repetitive questioning",
    "complaining": "Verbally non-aggressive: complaining",
    "negativism": "Verbally non-aggressive: negativism/refusal",
    "screaming": "Verbally agitated: screaming/shouting",
    "verbal aggression": "Verbally agitated: verbal aggression",
    "threatening": "Verbally agitated: threatening language",
    "hitting": "Physically aggressive: hitting",
    "kicking": "Physically aggressive: kicking",
    "pushing": "Physically aggressive: pushing",
}


def map_behaviours_to_cmai(behaviours: list[str]) -> list[dict[str, str]]:
    mappings = []
    for behaviour in behaviours:
        category = next((label for key, label in CMAI_BEHAVIOUR_MAP.items() if key in behaviour.lower()), "Unmapped observable behaviour (review required)")
        mappings.append({"behaviour": behaviour, "cmai_category": category})
    return mappings


def compute_final_score(acoustic_score: float, linguistic_score: float) -> float:
    acoustic = _bounded_score(acoustic_score, "acoustic_score")
    return round(0.4 * acoustic + 0.6 * _bounded_score(linguistic_score, "linguistic_score"), 4)


def analyze_person3(transcript: str | None, acoustic_features: Mapping[str, Any] | None, acoustic_score: float | None = None, analyzer: GeminiBehaviourAnalyzer | None = None) -> dict[str, Any]:
    resolved_acoustic_score = resolve_acoustic_score(acoustic_score, acoustic_features)
    gemini = (analyzer or GeminiBehaviourAnalyzer()).analyze(transcript, acoustic_features, resolved_acoustic_score)
    return {"acoustic_score": resolved_acoustic_score, "gemini": gemini, "final_score": compute_final_score(resolved_acoustic_score, gemini["agitation_score"]), "cmai_mapping": map_behaviours_to_cmai(gemini["behaviours"])}

"""Shared typed dataclasses for the audio agitation pipeline.

All modules import from here to avoid circular imports and ensure
a single source of truth for inter-module data contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any


@dataclass
class LatencyTrace:
    """Monotonic timestamps for end-to-end audio pipeline latency diagnostics."""

    microphone_ts: float | None = None
    queue_ts: float | None = None
    transcription_input_ts: float | None = None
    transcript_ts: float | None = None
    feature_extraction_ts: float | None = None
    inference_ts: float | None = None
    dashboard_render_ts: float | None = None

    def mark(self, stage: str) -> None:
        setattr(self, f"{stage}_ts", time.monotonic())

    def durations_ms(self) -> dict[str, float]:
        stages = [
            ("microphone", self.microphone_ts),
            ("queue", self.queue_ts),
            ("transcription_input", self.transcription_input_ts),
            ("transcript", self.transcript_ts),
            ("feature_extraction", self.feature_extraction_ts),
            ("inference", self.inference_ts),
            ("dashboard_render", self.dashboard_render_ts),
        ]
        out: dict[str, float] = {}
        previous_name: str | None = None
        previous_ts: float | None = None
        first_ts: float | None = None
        for name, ts in stages:
            if ts is None:
                continue
            if first_ts is None:
                first_ts = ts
            if previous_ts is not None and previous_name is not None:
                out[f"{previous_name}_to_{name}"] = round((ts - previous_ts) * 1000.0, 2)
            previous_name = name
            previous_ts = ts
        if first_ts is not None and previous_ts is not None:
            out["end_to_end"] = round((previous_ts - first_ts) * 1000.0, 2)
        return out


@dataclass
class BehaviourEvent:
    """Structured representation of a detected behaviour event."""

    event_id: str | None = None
    internal_code: str | None = None
    behaviour_type: str | None = None
    canonical_label: str = "Unmapped audio behaviour"
    cmai_category: str | None = None
    person: str | None = None
    timestamp: Any = None
    location: str | None = None
    severity: str | None = None
    duration: float | None = None
    trigger: str | None = None
    intervention: str | None = None
    outcome: str | None = None
    notes: str | None = None
    modality: str = "audio"
    raw_detected_behaviour: str | None = None
    mapping_status: str = "review_required"


# ---------------------------------------------------------------------------
# Acoustic branch
# ---------------------------------------------------------------------------

@dataclass
class AcousticFeatureWindow:
    """Timestamped feature snapshot for a 2-second audio window."""

    start_time: float   # Unix timestamp (seconds)
    end_time: float     # Unix timestamp (seconds)

    # Energy
    rms_mean: float = 0.0
    rms_max: float = 0.0
    rms_slope: float = 0.0          # linear regression slope over the window

    # Pitch (voiced frames only)
    pitch_median: float = 0.0
    pitch_range: float = 0.0        # max − min of voiced F0
    pitch_variance: float = 0.0

    # Spectral
    zcr_mean: float = 0.0
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    harmonic_to_noise_ratio: float = 0.0

    # Voice activity (Silero mask — not a gate)
    voiced_ratio: float = 0.0       # proportion of frames flagged as speech
    pause_ratio: float = 0.0        # 1 − voiced_ratio
    clipping_ratio: float = 0.0     # frames where |sample| >= 0.99

    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Transcript branch
# ---------------------------------------------------------------------------

@dataclass
class CommittedLine:
    """A transcript segment that local transcriber has confirmed will not change."""

    text: str
    timestamp: float    # Unix timestamp when line was committed
    latency_trace: LatencyTrace | None = None


@dataclass
class Utterance:
    """A completed, finalized unit of speech assembled by the UtteranceAggregator."""

    lines: list[CommittedLine]
    start_time: float   # Unix timestamp of first word
    end_time: float     # Unix timestamp of last committed line
    latency_trace: LatencyTrace | None = None

    @property
    def full_text(self) -> str:
        return " ".join(line.text.strip() for line in self.lines if line.text.strip())

    def duration(self) -> float:
        return self.end_time - self.start_time


# ---------------------------------------------------------------------------
# Linguistic features
# ---------------------------------------------------------------------------

@dataclass
class LinguisticFeatures:
    """Output of the linguistic analysis for one utterance."""

    repetition_score: float = 0.0
    question_repetition_score: float = 0.0
    negative_sentiment: float = 0.0
    urgency_score: float = 0.0
    threat_score: float = 0.0
    profanity_score: float = 0.0
    imperative_score: float = 0.0
    yelling_score: float = 0.0
    sexual_advance_score: float = 0.0
    complaint_score: float = 0.0
    negativism_score: float = 0.0

    # Raw evidence strings (for explainability panel)
    evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fused output
# ---------------------------------------------------------------------------

@dataclass
class FusedResult:
    """Final per-utterance output passed to the dashboard."""

    # Scores
    acoustic_score: float = 0.0         # 0–1, sigmoid-normalised
    linguistic_score: float = 0.0       # 0–1, linear-weighted
    raw_final_score: float = 0.0        # pre-smoothing fusion
    smoothed_score: float = 0.0         # EMA-smoothed display value

    # Metadata
    severity: str = "Low"               # Low / Mild / Moderate / High
    reliability: float = 1.0            # 0–1

    # Multi-label behaviour output
    behaviours: list[str] = field(default_factory=list)
    behaviour_events: list[BehaviourEvent] = field(default_factory=list)

    # Per-feature contributions for the explainability panel
    acoustic_contributions: dict[str, float] = field(default_factory=dict)
    linguistic_contributions: dict[str, float] = field(default_factory=dict)

    # Source data for downstream consumers
    utterance: Utterance | None = None
    acoustic_features: AcousticFeatureWindow | None = None
    linguistic_features: LinguisticFeatures | None = None

    # Optional Gemini comparison result (disabled by default)
    gemini_result: dict[str, Any] | None = None

    # End-to-end latency diagnostics for the utterance lifecycle
    latency_trace: LatencyTrace | None = None

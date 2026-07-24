"""Shared typed dataclasses for the audio agitation pipeline.

All modules import from here to avoid circular imports and ensure
a single source of truth for inter-module data contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    """A transcript segment that WhisperLiveKit has confirmed will not change."""

    text: str
    timestamp: float    # Unix timestamp when line was committed


@dataclass
class Utterance:
    """A completed, finalized unit of speech assembled by the UtteranceAggregator."""

    lines: list[CommittedLine]
    start_time: float   # Unix timestamp of first word
    end_time: float     # Unix timestamp of last committed line

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

    # Per-feature contributions for the explainability panel
    acoustic_contributions: dict[str, float] = field(default_factory=dict)
    linguistic_contributions: dict[str, float] = field(default_factory=dict)

    # Source data for downstream consumers
    utterance: Utterance | None = None
    acoustic_features: AcousticFeatureWindow | None = None
    linguistic_features: LinguisticFeatures | None = None

    # Optional Gemini comparison result (disabled by default)
    gemini_result: dict[str, Any] | None = None

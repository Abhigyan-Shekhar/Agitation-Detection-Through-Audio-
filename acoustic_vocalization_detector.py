"""Lightweight acoustic detection for non-speech human vocalizations.

This is a deterministic heuristic, not a trained dataset model. It catches
the common "raw microphone moan/groan" case by looking for sustained,
stable, voiced energy that is not speech-like enough to be handled by ASR.
Dataset-derived transcript labels live in ``strange_noise_labels.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config

try:
    import librosa
except ImportError:
    librosa = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AcousticVocalizationResult:
    """Best acoustic non-speech vocalization candidate for one audio window."""

    label: str | None = None
    score: float = 0.0
    evidence: str = ""


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def detect_acoustic_vocalization(audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> AcousticVocalizationResult:
    """Detect sustained moan/groan-like non-speech vocalizations from raw audio."""
    if librosa is None or audio.size < int(sample_rate * 0.35):
        return AcousticVocalizationResult()

    y = np.asarray(audio, dtype=np.float32)
    if not np.any(np.isfinite(y)):
        return AcousticVocalizationResult()
    y = np.nan_to_num(y)

    rms = float(np.sqrt(np.mean(np.square(y))))
    peak = float(np.max(np.abs(y)))
    if rms < config.ACOUSTIC_VOCALIZATION_MIN_RMS or peak < config.ACOUSTIC_VOCALIZATION_MIN_PEAK:
        return AcousticVocalizationResult()

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C6"),
            sr=sample_rate,
            frame_length=1024,
            hop_length=256,
        )
    except Exception:  # noqa: BLE001
        return AcousticVocalizationResult()

    if f0 is None or voiced_flag is None:
        return AcousticVocalizationResult()

    finite_f0 = f0[np.isfinite(f0)]
    pitch_coverage = float(finite_f0.size / max(f0.size, 1))
    if finite_f0.size < 3 or pitch_coverage < config.ACOUSTIC_VOCALIZATION_MIN_PITCH_COVERAGE:
        return AcousticVocalizationResult()

    pitch_median = float(np.median(finite_f0))
    pitch_range = float(np.ptp(finite_f0))
    pitch_std = float(np.std(finite_f0))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=256)[0]))
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sample_rate, n_fft=512, hop_length=256)))

    stable_pitch = pitch_range <= config.ACOUSTIC_VOCALIZATION_MAX_PITCH_RANGE
    low_noise = zcr <= config.ACOUSTIC_VOCALIZATION_MAX_ZCR
    low_centroid = spectral_centroid <= config.ACOUSTIC_VOCALIZATION_MAX_CENTROID
    pitch_in_human_vocal_range = 70.0 <= pitch_median <= 420.0

    if not (stable_pitch and low_noise and low_centroid and pitch_in_human_vocal_range):
        return AcousticVocalizationResult()

    stability = 1.0 - _clamp(pitch_range / max(config.ACOUSTIC_VOCALIZATION_MAX_PITCH_RANGE, 1e-6))
    energy = _clamp((rms - config.ACOUSTIC_VOCALIZATION_MIN_RMS) / 0.12)
    coverage = _clamp(pitch_coverage)
    score = _clamp(0.45 + 0.25 * stability + 0.20 * coverage + 0.10 * energy)

    label = "groaning" if pitch_median < config.ACOUSTIC_VOCALIZATION_GROAN_MAX_HZ else "moaning"
    evidence = (
        f"acoustic {label} heuristic: rms={rms:.3f}, peak={peak:.3f}, "
        f"pitch_median={pitch_median:.1f}Hz, pitch_range={pitch_range:.1f}Hz, "
        f"pitch_std={pitch_std:.1f}Hz, pitch_coverage={pitch_coverage:.2f}, "
        f"zcr={zcr:.3f}, centroid={spectral_centroid:.0f}Hz"
    )
    return AcousticVocalizationResult(label=label, score=round(score, 3), evidence=evidence)

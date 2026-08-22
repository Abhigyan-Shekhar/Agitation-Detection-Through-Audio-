"""Central configuration for the audio agitation pipeline.

All tuneable constants and environment flags live here. Import this module
everywhere rather than scattering magic numbers across the codebase.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000        # Hz — Whisper and Silero both want 16 kHz
FRAME_SIZE: int = 512            # samples per sounddevice callback (~32 ms)
CHANNELS: int = 1
DTYPE: str = "float32"
AUDIO_INPUT_DEVICE: str | int | None = os.getenv("AUDIO_INPUT_DEVICE") or None

# ---------------------------------------------------------------------------
# Local transcription
# ---------------------------------------------------------------------------
TRANSCRIPTION_ENGINE: str = os.getenv("TRANSCRIPTION_ENGINE", "faster-whisper")
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE: str | None = os.getenv("WHISPER_LANGUAGE", "en") or None
TRANSCRIPTION_WINDOW_SECONDS: float = float(os.getenv("TRANSCRIPTION_WINDOW_SECONDS", "5"))
TRANSCRIPTION_INTERVAL_SECONDS: float = float(os.getenv("TRANSCRIPTION_INTERVAL_SECONDS", "1"))
TRANSCRIPTION_STOP_TIMEOUT_SECONDS: float = float(os.getenv("TRANSCRIPTION_STOP_TIMEOUT_SECONDS", "30"))
USE_GPU_IF_AVAILABLE: bool = os.getenv("USE_GPU_IF_AVAILABLE", "true").lower() == "true"

# Local speaker diarization. The ECAPA model is loaded lazily on the
# transcription worker, never in the sounddevice callback or Streamlit loop.
ENABLE_SPEAKER_DIARIZATION: bool = os.getenv("ENABLE_SPEAKER_DIARIZATION", "true").lower() == "true"
DIARIZATION_BACKEND: str = os.getenv("DIARIZATION_BACKEND", "speechbrain-ecapa")
DIARIZATION_MODEL: str = os.getenv("DIARIZATION_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
# SpeechBrain SpeakerRecognition uses 0.25 for ECAPA verification. Keep a
# small margin for short, live microphone segments while remaining tunable.
DIARIZATION_SIMILARITY_THRESHOLD: float = float(os.getenv("DIARIZATION_SIMILARITY_THRESHOLD", "0.22"))
DIARIZATION_MIN_SEGMENT_SECONDS: float = float(os.getenv("DIARIZATION_MIN_SEGMENT_SECONDS", "1.0"))
DIARIZATION_MAX_SPEAKERS: int = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "6"))

# ---------------------------------------------------------------------------
# Utterance aggregator
# ---------------------------------------------------------------------------
UTTERANCE_SILENCE_SEC: float = 1.8   # silence gap that finalises an utterance
MAX_UTTERANCE_SEC: float = 12.0      # hard cap — prevents endless accumulation

# ---------------------------------------------------------------------------
# Acoustic feature extraction
# ---------------------------------------------------------------------------
ACOUSTIC_WINDOW_SEC: float = 2.0    # length of each feature window
ACOUSTIC_HOP_SEC: float = 0.5       # hop between windows
AUDIO_RING_BUFFER_SEC: float = 60.0  # rolling audio history kept in memory
VAD_THRESHOLD: float = 0.5           # Silero speech-probability cut-off

# ---------------------------------------------------------------------------
# Baseline manager
# ---------------------------------------------------------------------------
BASELINE_COLLECT_MIN: float = 2.0    # minimum calm recording needed (minutes)
BASELINE_ROLLING_MIN: float = 5.0    # rolling fallback window (minutes)
# Robust baseline normalization: use median/percentile spread plus minimum
# tolerances so calm calibration does not create hair-trigger z-scores.
BASELINE_STD_REL_FLOOR: float = float(os.getenv("BASELINE_STD_REL_FLOOR", "0.35"))
BASELINE_RMS_STD_FLOOR: float = float(os.getenv("BASELINE_RMS_STD_FLOOR", "0.015"))
BASELINE_PEAK_STD_FLOOR: float = float(os.getenv("BASELINE_PEAK_STD_FLOOR", "0.04"))
BASELINE_PITCH_STD_FLOOR: float = float(os.getenv("BASELINE_PITCH_STD_FLOOR", "35"))
BASELINE_ZCR_STD_FLOOR: float = float(os.getenv("BASELINE_ZCR_STD_FLOOR", "0.015"))
BASELINE_CENTROID_STD_FLOOR: float = float(os.getenv("BASELINE_CENTROID_STD_FLOOR", "250"))
# The dashboard has no resident identifier yet, so this is intentionally a
# local per-deployment file. Set BASELINE_STORAGE_PATH per resident/device.
BASELINE_STORAGE_PATH: str = os.getenv("BASELINE_STORAGE_PATH", ".odu_personal_baseline.json")

# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------
Z_CLIP: float = 3.0                  # clamp Z-scores to ±3 before fusion
EMA_ALPHA_UP: float = 0.55           # fast escalation
EMA_ALPHA_DOWN: float = 0.20         # slow de-escalation

# Acoustic sigmoid bias (see score_fusion.py module docstring).
# Subtract this from the weighted Z-sum before passing through sigmoid so that
# all-zero Z-scores (no personal baseline) map to ~0.047 instead of 0.5.
# Increase to suppress the acoustic branch further; decrease toward 0 to
# revert to the pre-bias behaviour (set to 0 to disable).
ACOUSTIC_SIGMOID_BIAS: float = float(os.getenv("ACOUSTIC_SIGMOID_BIAS", "3.0"))

# Acoustic branch weights (must sum to 1.0)
ACOUSTIC_WEIGHTS: dict[str, float] = {
    "energy_z":           0.30,
    "energy_burst_z":     0.20,
    "pitch_range_z":      0.15,
    "pitch_variance_z":   0.10,
    "speech_rate_z":      0.15,
    "pause_irregularity_z": 0.10,
}

# Linguistic branch weights (must sum to 1.0)
LINGUISTIC_WEIGHTS: dict[str, float] = {
    "repetition_score":          0.24,
    "question_repetition_score": 0.18,
    "negative_sentiment":        0.15,
    "urgency_score":             0.15,
    "threat_score":              0.15,
    "profanity_score":           0.05,
    "sexual_advance_score":      0.04,
    "strange_noise_score":       0.04,
}

# Final fusion
ACOUSTIC_FUSION_WEIGHT: float = 0.60
LINGUISTIC_FUSION_WEIGHT: float = 0.40

# Behaviour thresholds
BEHAVIOUR_REPETITION_THRESHOLD: float = 0.65
BEHAVIOUR_Q_REP_THRESHOLD: float = 0.70
BEHAVIOUR_REQUEST_REP_THRESHOLD: float = 0.65
BEHAVIOUR_ENERGY_Z_SHOUT: float = float(os.getenv("BEHAVIOUR_ENERGY_Z_SHOUT", "2.5"))
BEHAVIOUR_ENERGY_BURST_SHOUT: float = float(os.getenv("BEHAVIOUR_ENERGY_BURST_SHOUT", "1.8"))
SCREAM_ON_SCORE_THRESHOLD: float = float(os.getenv("SCREAM_ON_SCORE_THRESHOLD", "0.72"))
SCREAM_OFF_SCORE_THRESHOLD: float = float(os.getenv("SCREAM_OFF_SCORE_THRESHOLD", "0.45"))
SCREAM_MIN_CONSECUTIVE_WINDOWS: int = int(os.getenv("SCREAM_MIN_CONSECUTIVE_WINDOWS", "3"))
SCREAM_RECOVERY_CONSECUTIVE_WINDOWS: int = int(os.getenv("SCREAM_RECOVERY_CONSECUTIVE_WINDOWS", "2"))
SCREAM_MIN_DURATION_SEC: float = float(os.getenv("SCREAM_MIN_DURATION_SEC", "1.5"))
BEHAVIOUR_ABSOLUTE_RMS_SHOUT: float = float(os.getenv("BEHAVIOUR_ABSOLUTE_RMS_SHOUT", "0.18"))
BEHAVIOUR_ABSOLUTE_PEAK_SHOUT: float = float(os.getenv("BEHAVIOUR_ABSOLUTE_PEAK_SHOUT", "0.65"))
BEHAVIOUR_CLIPPING_SHOUT: float = float(os.getenv("BEHAVIOUR_CLIPPING_SHOUT", "0.02"))
BEHAVIOUR_VERBAL_AGGR_ACOUSTIC: float = 0.65
BEHAVIOUR_VERBAL_AGGR_SENTIMENT: float = 0.50
BEHAVIOUR_VERBAL_AGGR_THREAT: float = 0.45
BEHAVIOUR_URGENCY_THRESHOLD: float = 0.60
BEHAVIOUR_URGENCY_ACOUSTIC: float = 0.50
BEHAVIOUR_COMPLAINT_THRESHOLD: float = 0.55
BEHAVIOUR_NEGATIVISM_THRESHOLD: float = 0.55
BEHAVIOUR_STRANGE_NOISE_THRESHOLD: float = 0.60

# Strange-noise event segmentation: keep one continuous non-speech
# vocalization as one behavioural episode across overlapping windows.
STRANGE_NOISE_OFF_CONSECUTIVE_WINDOWS: int = int(os.getenv("STRANGE_NOISE_OFF_CONSECUTIVE_WINDOWS", "2"))
STRANGE_NOISE_EVENT_COOLDOWN_SEC: float = float(os.getenv("STRANGE_NOISE_EVENT_COOLDOWN_SEC", "1.0"))
ACOUSTIC_VOCALIZATION_MIN_RMS: float = float(os.getenv("ACOUSTIC_VOCALIZATION_MIN_RMS", "0.025"))
ACOUSTIC_VOCALIZATION_MIN_PEAK: float = float(os.getenv("ACOUSTIC_VOCALIZATION_MIN_PEAK", "0.06"))
ACOUSTIC_VOCALIZATION_MIN_PITCH_COVERAGE: float = float(os.getenv("ACOUSTIC_VOCALIZATION_MIN_PITCH_COVERAGE", "0.35"))
ACOUSTIC_VOCALIZATION_MAX_PITCH_RANGE: float = float(os.getenv("ACOUSTIC_VOCALIZATION_MAX_PITCH_RANGE", "120"))
ACOUSTIC_VOCALIZATION_MAX_ZCR: float = float(os.getenv("ACOUSTIC_VOCALIZATION_MAX_ZCR", "0.12"))
ACOUSTIC_VOCALIZATION_MAX_CENTROID: float = float(os.getenv("ACOUSTIC_VOCALIZATION_MAX_CENTROID", "1800"))
ACOUSTIC_VOCALIZATION_GROAN_MAX_HZ: float = float(os.getenv("ACOUSTIC_VOCALIZATION_GROAN_MAX_HZ", "170"))

# Severity thresholds
SEVERITY_LOW_MAX: float = 0.35
SEVERITY_MILD_MAX: float = 0.60
SEVERITY_MODERATE_MAX: float = 0.80

# Reliability penalties
RELIABILITY_CLIPPING_PENALTY: float = 0.25
RELIABILITY_NOISE_PENALTY: float = 0.20
RELIABILITY_SHORT_UTTERANCE_PENALTY: float = 0.20
RELIABILITY_BRANCH_DISAGREEMENT_PENALTY: float = 0.20
RELIABILITY_MISSING_BASELINE_PENALTY: float = 0.15

# ---------------------------------------------------------------------------
# Transcript history
# ---------------------------------------------------------------------------
TRANSCRIPT_HISTORY_SEC: float = 60.0   # rolling window for linguistic analysis

# ---------------------------------------------------------------------------
# Analysis mode
# ---------------------------------------------------------------------------
ANALYSIS_MODE: str = os.getenv("ANALYSIS_MODE", "rule_based")
ENABLE_GEMINI_COMPARISON: bool = (
    os.getenv("ENABLE_GEMINI_COMPARISON", "false").lower() == "true"
)

# ---------------------------------------------------------------------------
# Debug / diagnostic logging
# ---------------------------------------------------------------------------
# When true, the score_fusion and behaviour_classifier loggers are set to
# DEBUG so that every intermediate value (baseline stats, z-scores, acoustic
# score, scream gate flags, final label) is emitted to the log.
# Enable via env var: DEBUG_TRACE_LOGGING=true
DEBUG_TRACE_LOGGING: bool = os.getenv("DEBUG_TRACE_LOGGING", "false").lower() == "true"

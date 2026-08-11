"""Central configuration for the audio agitation pipeline.

All tuneable constants and environment flags live here. Import this module
everywhere rather than scattering magic numbers across the codebase.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import sys
from urllib.parse import urlencode


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000        # Hz — Whisper and Silero both want 16 kHz
FRAME_SIZE: int = 512            # samples per sounddevice callback (~32 ms)
CHANNELS: int = 1
DTYPE: str = "float32"
AUDIO_INPUT_DEVICE: str | int | None = os.getenv("AUDIO_INPUT_DEVICE") or None

# ---------------------------------------------------------------------------
# Transcription / WhisperLiveKit
# ---------------------------------------------------------------------------
_WLK_PLATFORM_DEFAULT = (
    sys.version_info < (3, 13)
    and platform.system() != "Darwin"
    and importlib.util.find_spec("nemo") is not None
)
ENABLE_SPEAKER_DIARIZATION: bool = _env_bool(
    "ENABLE_SPEAKER_DIARIZATION", _WLK_PLATFORM_DEFAULT
)
DIARIZATION_BACKEND: str = os.getenv("DIARIZATION_BACKEND", "sortformer")
# Current WLK Sortformer does not expose a maximum-speaker CLI option. Keep this
# as an extension point without passing an unsupported argument to the server.
MAX_SPEAKERS: int | None = (
    int(value) if (value := os.getenv("MAX_SPEAKERS", "").strip()) else None
)
SPEAKER_ALIASES_JSON: str = os.getenv("SPEAKER_ALIASES_JSON", "")

TRANSCRIPTION_ENGINE: str = os.getenv(
    "TRANSCRIPTION_ENGINE",
    "whisperlivekit",
).strip().lower()
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE: str | None = os.getenv("WHISPER_LANGUAGE", "en") or None
TRANSCRIPTION_WINDOW_SECONDS: float = float(os.getenv("TRANSCRIPTION_WINDOW_SECONDS", "5"))
TRANSCRIPTION_INTERVAL_SECONDS: float = float(os.getenv("TRANSCRIPTION_INTERVAL_SECONDS", "1"))
TRANSCRIPTION_STOP_TIMEOUT_SECONDS: float = float(os.getenv("TRANSCRIPTION_STOP_TIMEOUT_SECONDS", "30"))
USE_GPU_IF_AVAILABLE: bool = os.getenv("USE_GPU_IF_AVAILABLE", "true").lower() == "true"

WLK_HOST: str = os.getenv("WLK_HOST", "127.0.0.1")
WLK_PORT: int = int(os.getenv("WLK_PORT", "8000"))
WLK_PATH: str = os.getenv("WLK_PATH", "/asr")
WLK_OUTPUT_MODE: str = os.getenv("WLK_OUTPUT_MODE", "diff")
_WLK_QUERY = urlencode({"mode": WLK_OUTPUT_MODE}) if WLK_OUTPUT_MODE else ""
WLK_URL: str = (
    f"ws://{WLK_HOST}:{WLK_PORT}{WLK_PATH}"
    f"{'&' if '?' in WLK_PATH else '?'}{_WLK_QUERY}"
    if _WLK_QUERY
    else f"ws://{WLK_HOST}:{WLK_PORT}{WLK_PATH}"
)
WLK_BACKEND: str = os.getenv("WLK_BACKEND", "auto")
WLK_AUTO_LAUNCH: bool = _env_bool("WLK_AUTO_LAUNCH", True)
WLK_STARTUP_TIMEOUT_SEC: float = float(os.getenv("WLK_STARTUP_TIMEOUT_SEC", "180"))

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

# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------
Z_CLIP: float = 3.0                  # clamp Z-scores to ±3 before fusion
EMA_ALPHA_UP: float = 0.55           # fast escalation
EMA_ALPHA_DOWN: float = 0.20         # slow de-escalation

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
BEHAVIOUR_ENERGY_Z_SHOUT: float = 2.0
BEHAVIOUR_ENERGY_BURST_SHOUT: float = 0.70
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

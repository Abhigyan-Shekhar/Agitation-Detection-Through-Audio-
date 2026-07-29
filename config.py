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
# WhisperLiveKit server
# ---------------------------------------------------------------------------
WLK_HOST: str = os.getenv("WLK_HOST", "127.0.0.1")
WLK_PORT: int = int(os.getenv("WLK_PORT", "8000"))
WLK_PATH: str = os.getenv("WLK_PATH", "/asr")
WLK_URL: str = f"ws://{WLK_HOST}:{WLK_PORT}{WLK_PATH}"

# WLK model settings (used when auto-launching the server)
WLK_MODEL: str = os.getenv("WLK_MODEL", "small")
WLK_LANGUAGE: str = os.getenv("WLK_LANGUAGE", "auto")
WLK_BACKEND: str = os.getenv("WLK_BACKEND", "faster-whisper")
# If True, dashboard.py will spawn wlk as a subprocess automatically
WLK_AUTO_LAUNCH: bool = os.getenv("WLK_AUTO_LAUNCH", "true").lower() == "true"

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
    "repetition_score":          0.30,
    "question_repetition_score": 0.20,
    "negative_sentiment":        0.15,
    "urgency_score":             0.15,
    "threat_score":              0.15,
    "profanity_score":           0.05,
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
BEHAVIOUR_VERBAL_AGGR_ACOUSTIC: float = 0.65
BEHAVIOUR_VERBAL_AGGR_SENTIMENT: float = 0.50
BEHAVIOUR_VERBAL_AGGR_THREAT: float = 0.45
BEHAVIOUR_URGENCY_THRESHOLD: float = 0.60
BEHAVIOUR_URGENCY_ACOUSTIC: float = 0.50

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

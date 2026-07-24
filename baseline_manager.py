"""Patient-specific baseline manager.

Responsibilities
----------------
* Collect ``AcousticFeatureWindow`` samples during a calm baseline
  recording period triggered by the user from the dashboard.
* Compute ``mean`` and ``std`` per acoustic feature from the collected
  samples once sufficient data has been gathered.
* Expose ``z_score(feature_name, value)`` for feature normalization in
  score_fusion.py.
* Fall back to a **rolling baseline** (last ``BASELINE_ROLLING_MIN``
  minutes of windows) when no personal baseline exists, and flag
  results as lower confidence.

Design notes
------------
* Thread-safe: all mutable state is protected by a ``threading.Lock``.
* Persists nothing to disk in this version (future: JSON sidecar).
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Deque

import numpy as np

import config
from event_models import AcousticFeatureWindow

logger = logging.getLogger(__name__)

# Features that are z-scored for the fusion model
_FEATURE_NAMES: tuple[str, ...] = (
    "rms_mean",
    "rms_max",
    "rms_slope",
    "pitch_median",
    "pitch_range",
    "pitch_variance",
    "zcr_mean",
    "spectral_centroid",
    "voiced_ratio",
    "pause_ratio",
)

# Minimum number of windows before a personal baseline is considered valid
_MIN_WINDOWS_FOR_PERSONAL: int = int(
    (config.BASELINE_COLLECT_MIN * 60) / config.ACOUSTIC_HOP_SEC
)


class BaselineManager:
    """Manages personal and rolling acoustic baselines for z-score normalisation.

    Parameters
    ----------
    rolling_window_min:
        Duration of the rolling fallback baseline in minutes.
    """

    def __init__(self, rolling_window_min: float = config.BASELINE_ROLLING_MIN) -> None:
        self._lock = threading.Lock()

        # Personal baseline (set during explicit calibration)
        self._personal_mean: dict[str, float] | None = None
        self._personal_std: dict[str, float] | None = None
        self._personal_n: int = 0

        # Calibration mode
        self._calibrating: bool = False
        self._calibration_samples: list[AcousticFeatureWindow] = []
        self._calibration_start: float | None = None

        # Rolling fallback: bounded deque keyed on timestamp
        max_rolling = int((rolling_window_min * 60) / config.ACOUSTIC_HOP_SEC) + 1
        self._rolling: Deque[AcousticFeatureWindow] = deque(maxlen=max_rolling)

    # ------------------------------------------------------------------
    # Calibration API (called from dashboard)
    # ------------------------------------------------------------------

    def start_calibration(self) -> None:
        """Begin collecting personal baseline samples."""
        with self._lock:
            self._calibrating = True
            self._calibration_samples.clear()
            self._calibration_start = time.time()
        logger.info("Baseline calibration started")

    def stop_calibration(self) -> bool:
        """Finalise calibration. Returns True if enough data was collected."""
        with self._lock:
            self._calibrating = False
            samples = list(self._calibration_samples)

        if len(samples) < _MIN_WINDOWS_FOR_PERSONAL:
            elapsed = (
                time.time() - (self._calibration_start or time.time())
            ) / 60.0
            logger.warning(
                "Calibration stopped early — only %d windows collected "
                "(need %d, ~%.1f min minimum).",
                len(samples),
                _MIN_WINDOWS_FOR_PERSONAL,
                config.BASELINE_COLLECT_MIN,
            )
            return False

        mean, std = self._compute_stats(samples)
        with self._lock:
            self._personal_mean = mean
            self._personal_std = std
            self._personal_n = len(samples)

        logger.info(
            "Baseline calibration complete — %d windows, personal baseline set",
            len(samples),
        )
        return True

    def reset_calibration(self) -> None:
        """Clear the personal baseline and revert to rolling fallback."""
        with self._lock:
            self._personal_mean = None
            self._personal_std = None
            self._personal_n = 0
            self._calibration_samples.clear()
        logger.info("Personal baseline reset")

    @property
    def is_calibrating(self) -> bool:
        with self._lock:
            return self._calibrating

    @property
    def has_personal_baseline(self) -> bool:
        with self._lock:
            return self._personal_mean is not None

    @property
    def calibration_window_count(self) -> int:
        with self._lock:
            return len(self._calibration_samples)

    @property
    def calibration_progress(self) -> float:
        """Returns 0.0 – 1.0 progress toward minimum required windows."""
        with self._lock:
            return min(1.0, len(self._calibration_samples) / max(_MIN_WINDOWS_FOR_PERSONAL, 1))

    # ------------------------------------------------------------------
    # Feed API (called by AcousticWorker on each new window)
    # ------------------------------------------------------------------

    def feed(self, window: AcousticFeatureWindow) -> None:
        """Accept a new feature window. Always updates the rolling baseline."""
        with self._lock:
            self._rolling.append(window)
            if self._calibrating:
                self._calibration_samples.append(window)

    # ------------------------------------------------------------------
    # Z-score API (called by score_fusion.py)
    # ------------------------------------------------------------------

    def z_score(self, feature_name: str, value: float) -> float:
        """Return the Z-score of ``value`` relative to the active baseline.

        Clips to ±``config.Z_CLIP`` before returning.
        Falls back to 0.0 if standard deviation is zero or data is unavailable.
        """
        mean, std = self._active_baseline(feature_name)
        if mean is None or std is None or std < 1e-9:
            return 0.0
        z = (value - mean) / std
        return float(np.clip(z, -config.Z_CLIP, config.Z_CLIP))

    def missing_baseline_penalty(self) -> float:
        """Reliability penalty (0–1) when no personal baseline exists."""
        return config.RELIABILITY_MISSING_BASELINE_PENALTY if not self.has_personal_baseline else 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_baseline(self, feature_name: str) -> tuple[float | None, float | None]:
        with self._lock:
            if self._personal_mean and feature_name in self._personal_mean:
                return (
                    self._personal_mean[feature_name],
                    self._personal_std.get(feature_name),  # type: ignore[union-attr]
                )
            # Rolling fallback
            rolling = list(self._rolling)

        if len(rolling) < 3:
            return None, None

        values = [getattr(w, feature_name, None) for w in rolling]
        values = [v for v in values if v is not None and math.isfinite(v)]
        if len(values) < 3:
            return None, None
        return float(np.mean(values)), float(np.std(values))

    @staticmethod
    def _compute_stats(
        samples: list[AcousticFeatureWindow],
    ) -> tuple[dict[str, float], dict[str, float]]:
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for feat in _FEATURE_NAMES:
            values = [
                getattr(s, feat)
                for s in samples
                if getattr(s, feat, None) is not None
                and math.isfinite(getattr(s, feat))
            ]
            if values:
                mean[feat] = float(np.mean(values))
                std[feat] = float(np.std(values)) if len(values) > 1 else 1.0
            else:
                mean[feat] = 0.0
                std[feat] = 1.0
        return mean, std

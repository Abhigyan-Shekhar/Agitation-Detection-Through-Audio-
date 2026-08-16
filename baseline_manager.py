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
        self._personal_median: dict[str, float] | None = None
        self._personal_p10: dict[str, float] | None = None
        self._personal_p90: dict[str, float] | None = None
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

        mean, std, median, p10, p90 = self._compute_stats(samples)
        with self._lock:
            self._personal_mean = mean
            self._personal_std = std
            self._personal_median = median
            self._personal_p10 = p10
            self._personal_p90 = p90
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
            self._personal_median = None
            self._personal_p10 = None
            self._personal_p90 = None
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
    def minimum_windows_for_personal(self) -> int:
        return _MIN_WINDOWS_FOR_PERSONAL

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _FEATURE_NAMES

    def personal_baseline_stats(self) -> dict[str, tuple[float, float]]:
        """Return a thread-safe copy of personal baseline mean/std values."""
        with self._lock:
            if self._personal_mean is None or self._personal_std is None:
                return {}
            return {
                feat: (
                    self._personal_mean.get(feat, 0.0),
                    self._personal_std.get(feat, 0.0),
                )
                for feat in _FEATURE_NAMES
            }


    def personal_baseline_summary(self) -> dict[str, dict[str, float]]:
        """Return robust personal baseline distribution statistics for debugging."""
        with self._lock:
            if not all((self._personal_mean, self._personal_std, self._personal_median, self._personal_p10, self._personal_p90)):
                return {}
            return {
                feat: {
                    "mean": self._personal_mean.get(feat, 0.0),
                    "std": self._personal_std.get(feat, 0.0),
                    "median": self._personal_median.get(feat, 0.0),
                    "p10": self._personal_p10.get(feat, 0.0),
                    "p90": self._personal_p90.get(feat, 0.0),
                }
                for feat in _FEATURE_NAMES
            }

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
                count = len(self._calibration_samples)
            else:
                count = 0
        if count and (count == 1 or count % 10 == 0 or count >= _MIN_WINDOWS_FOR_PERSONAL):
            logger.info(
                "BaselineManager.feed collected calibration window %d/%d "
                "(manager_id=%s)",
                count,
                _MIN_WINDOWS_FOR_PERSONAL,
                id(self),
            )

    # ------------------------------------------------------------------
    # Z-score API (called by score_fusion.py)
    # ------------------------------------------------------------------

    def z_score(self, feature_name: str, value: float) -> float:
        """Return the Z-score of ``value`` relative to the active baseline.

        Clips to ±``config.Z_CLIP`` before returning.
        Falls back to 0.0 if standard deviation is zero or data is unavailable.
        """
        center, std = self._active_baseline(feature_name)
        if center is None or std is None or std < 1e-9:
            return 0.0
        z = (value - center) / std
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
                center = (self._personal_median or self._personal_mean).get(feature_name, self._personal_mean[feature_name])
                return (
                    center,
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
        return float(np.median(values)), self._robust_std(feature_name, values)

    @staticmethod
    def _compute_stats(
        samples: list[AcousticFeatureWindow],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        median: dict[str, float] = {}
        p10: dict[str, float] = {}
        p90: dict[str, float] = {}
        for feat in _FEATURE_NAMES:
            values = [
                getattr(s, feat)
                for s in samples
                if getattr(s, feat, None) is not None
                and math.isfinite(getattr(s, feat))
            ]
            if values:
                mean[feat] = float(np.mean(values))
                median[feat] = float(np.median(values))
                p10[feat] = float(np.percentile(values, 10))
                p90[feat] = float(np.percentile(values, 90))
                std[feat] = BaselineManager._robust_std(feat, values)
            else:
                mean[feat] = 0.0
                median[feat] = 0.0
                p10[feat] = 0.0
                p90[feat] = 0.0
                std[feat] = 1.0
        return mean, std, median, p10, p90

    @staticmethod
    def _robust_std(feature_name: str, values: list[float]) -> float:
        """Estimate spread from percentiles and enforce feature-specific tolerance floors.

        Calm calibration can be very consistent; using its tiny raw standard
        deviation makes ordinary speech variation look like an extreme event.
        This keeps the personal baseline, but treats it as a normal range rather
        than a single narrow boundary.
        """
        arr = np.asarray(values, dtype=float)
        raw_std = float(np.std(arr)) if arr.size > 1 else 0.0
        iqr_std = float((np.percentile(arr, 75) - np.percentile(arr, 25)) / 1.349) if arr.size > 1 else 0.0
        p80_std = float((np.percentile(arr, 90) - np.percentile(arr, 10)) / 2.563) if arr.size > 1 else 0.0
        center = abs(float(np.median(arr))) if arr.size else 0.0
        rel_floor = center * config.BASELINE_STD_REL_FLOOR
        absolute_floors = {
            "rms_mean": config.BASELINE_RMS_STD_FLOOR,
            "rms_max": config.BASELINE_PEAK_STD_FLOOR,
            "pitch_median": config.BASELINE_PITCH_STD_FLOOR,
            "pitch_range": config.BASELINE_PITCH_STD_FLOOR,
            "pitch_variance": config.BASELINE_PITCH_STD_FLOOR ** 2,
            "zcr_mean": config.BASELINE_ZCR_STD_FLOOR,
            "spectral_centroid": config.BASELINE_CENTROID_STD_FLOOR,
        }
        return max(raw_std, iqr_std, p80_std, rel_floor, absolute_floors.get(feature_name, 1e-3))

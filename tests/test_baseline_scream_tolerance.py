import pytest

import config
from baseline_manager import BaselineManager
from behaviour_classifier import BehaviourClassifier
from event_models import AcousticFeatureWindow, FusedResult, LinguisticFeatures


def _window(t: float, rms: float, peak: float | None = None, pitch_range: float = 45.0) -> AcousticFeatureWindow:
    return AcousticFeatureWindow(
        start_time=t,
        end_time=t + config.ACOUSTIC_WINDOW_SEC,
        rms_mean=rms,
        rms_max=peak if peak is not None else rms * 2.0,
        pitch_median=180.0,
        pitch_range=pitch_range,
        pitch_variance=pitch_range * 10.0,
        zcr_mean=0.05,
        spectral_centroid=1600.0,
        voiced_ratio=0.75,
        pause_ratio=0.25,
    )


def _result(acoustic: AcousticFeatureWindow, acoustic_score: float, energy: float, burst: float) -> FusedResult:
    return FusedResult(
        acoustic_score=acoustic_score,
        linguistic_score=0.0,
        raw_final_score=acoustic_score,
        smoothed_score=acoustic_score,
        severity="Moderate",
        reliability=1.0,
        acoustic_features=acoustic,
        linguistic_features=LinguisticFeatures(),
        acoustic_contributions={"energy_above_baseline": energy, "energy_burst": burst},
        linguistic_contributions={},
    )


def test_robust_baseline_tolerates_slightly_louder_normal_speech():
    bm = BaselineManager()
    bm.start_calibration()
    for i in range(bm.minimum_windows_for_personal):
        bm.feed(_window(float(i), rms=0.040 + (0.001 if i % 2 else 0.0), peak=0.11))
    assert bm.stop_calibration()

    # 50% louder than calibration is raised conversational speech, not an
    # immediate scream. The robust floor keeps the z-score inside tolerance.
    assert bm.z_score("rms_mean", 0.060) < config.BEHAVIOUR_ENERGY_Z_SHOUT


def test_screaming_requires_temporal_persistence_and_recovers_with_hysteresis():
    classifier = BehaviourClassifier()
    base_t = 1_000.0

    # A single loud/spiky window is suppressed even though the instantaneous
    # acoustic evidence is high.
    first = classifier.classify(_result(_window(base_t, 0.30, 0.80), 0.95, 0.9, 0.5))
    assert "Screaming" not in first.behaviours

    second = classifier.classify(_result(_window(base_t + 0.5, 0.31, 0.82), 0.95, 0.9, 0.5))
    assert "Screaming" not in second.behaviours

    third = classifier.classify(_result(_window(base_t + 2.0, 0.32, 0.85), 0.95, 0.9, 0.5))
    assert "Screaming" in third.behaviours

    # Scores between OFF and ON hold the active state (hysteresis), then two
    # low recovery windows clear it.
    held = classifier.classify(_result(_window(base_t + 2.5, 0.20, 0.60), 0.50, 0.3, 0.2))
    assert "Screaming" in held.behaviours
    classifier.classify(_result(_window(base_t + 3.0, 0.05, 0.12), 0.20, 0.0, 0.0))
    recovered = classifier.classify(_result(_window(base_t + 3.5, 0.05, 0.12), 0.20, 0.0, 0.0))
    assert "Screaming" not in recovered.behaviours

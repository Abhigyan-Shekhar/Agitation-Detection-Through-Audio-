"""Tests for score fusion and temporal smoothing."""
from __future__ import annotations

import time
import math
import pytest

from baseline_manager import BaselineManager
from event_models import (
    AcousticFeatureWindow,
    CommittedLine,
    FusedResult,
    LinguisticFeatures,
    Utterance,
)
from score_fusion import ScoreFusion, _sigmoid, _clamp


def _make_utterance(text: str = "test") -> Utterance:
    ts = time.time()
    return Utterance(
        lines=[CommittedLine(text=text, timestamp=ts)],
        start_time=ts - 2.0,
        end_time=ts,
    )


def _make_acoustic(**kwargs) -> AcousticFeatureWindow:
    defaults = dict(
        start_time=time.time() - 2.0,
        end_time=time.time(),
        rms_mean=0.02,
        rms_max=0.05,
        rms_slope=0.0,
        pitch_median=120.0,
        pitch_range=20.0,
        pitch_variance=100.0,
        zcr_mean=0.05,
        spectral_centroid=2000.0,
        voiced_ratio=0.7,
        pause_ratio=0.3,
        clipping_ratio=0.0,
    )
    defaults.update(kwargs)
    return AcousticFeatureWindow(**defaults)


def _make_linguistic(**kwargs) -> LinguisticFeatures:
    defaults = dict(
        repetition_score=0.0,
        question_repetition_score=0.0,
        negative_sentiment=0.0,
        urgency_score=0.0,
        threat_score=0.0,
        profanity_score=0.0,
        imperative_score=0.0,
    )
    defaults.update(kwargs)
    return LinguisticFeatures(**defaults)


class TestHelpers:
    def test_sigmoid_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert _sigmoid(10.0) > 0.99

    def test_sigmoid_large_negative(self):
        assert _sigmoid(-10.0) < 0.01

    def test_clamp_within(self):
        assert _clamp(0.5) == pytest.approx(0.5)

    def test_clamp_above(self):
        assert _clamp(1.5) == pytest.approx(1.0)

    def test_clamp_below(self):
        assert _clamp(-0.5) == pytest.approx(0.0)


class TestScoreFusion:
    def setup_method(self):
        self.bm = BaselineManager()
        self.fusion = ScoreFusion(self.bm)

    def test_all_zero_inputs(self):
        acoustic = _make_acoustic(
            rms_mean=0.0, rms_max=0.0, rms_slope=0.0,
            pitch_median=0.0, pitch_range=0.0, pitch_variance=0.0,
            voiced_ratio=0.0, pause_ratio=0.0, clipping_ratio=0.0,
        )
        linguistic = _make_linguistic()
        result = self.fusion.fuse(_make_utterance(), acoustic, linguistic)
        # With all-zero features and no baseline, z-scores are 0.
        # After sigmoid bias (-3.0), acoustic_score ≈ sigmoid(-3) ≈ 0.047.
        # Raw fused score = 0.60 * 0.047 + 0.40 * 0.0 = 0.028 → well inside Low.
        assert 0.0 <= result.acoustic_score <= 0.10, (
            f"Expected near-zero acoustic score with no baseline and all-zero inputs, "
            f"got {result.acoustic_score:.4f}"
        )
        assert result.linguistic_score == pytest.approx(0.0)

    def test_scores_bounded(self):
        for _ in range(5):
            result = self.fusion.fuse(
                _make_utterance(),
                _make_acoustic(rms_mean=0.5, voiced_ratio=0.9),
                _make_linguistic(urgency_score=0.8, threat_score=0.7),
            )
            assert 0.0 <= result.acoustic_score <= 1.0
            assert 0.0 <= result.linguistic_score <= 1.0
            assert 0.0 <= result.raw_final_score <= 1.0
            assert 0.0 <= result.smoothed_score <= 1.0

    def test_ema_escalates_fast(self):
        """Smoothed score should increase quickly on escalation."""
        self.fusion.reset()
        results = []
        for i in range(5):
            r = self.fusion.fuse(
                _make_utterance(),
                _make_acoustic(rms_mean=0.5),
                _make_linguistic(urgency_score=0.9, threat_score=0.8),
            )
            results.append(r.smoothed_score)
        # Escalating signal → smoothed score should be rising
        assert results[-1] >= results[0]

    def test_ema_de_escalates_slow(self):
        """After a high score, low signal should decrease it slowly."""
        # Prime with high score
        for _ in range(3):
            self.fusion.fuse(
                _make_utterance(),
                _make_acoustic(rms_mean=0.5),
                _make_linguistic(urgency_score=0.9, threat_score=0.9),
            )
        prev = self.fusion._prev_smoothed

        # One calm utterance
        calm = self.fusion.fuse(
            _make_utterance(),
            _make_acoustic(rms_mean=0.01),
            _make_linguistic(),
        )
        # Should not drop more than EMA_ALPHA_DOWN * prev
        assert calm.smoothed_score > prev * 0.5

    def test_severity_levels(self):
        self.fusion.reset()
        r_calm = self.fusion.fuse(
            _make_utterance(), _make_acoustic(), _make_linguistic()
        )
        # First call will be near baseline → Low or Mild
        assert r_calm.severity in ("Low", "Mild", "Moderate", "High")

    def test_reliability_decreases_with_clipping(self):
        r_clean = self.fusion.fuse(
            _make_utterance(), _make_acoustic(clipping_ratio=0.0), _make_linguistic()
        )
        r_clipped = self.fusion.fuse(
            _make_utterance(), _make_acoustic(clipping_ratio=0.8), _make_linguistic()
        )
        assert r_clipped.reliability <= r_clean.reliability

    def test_contributions_dict_not_empty(self):
        result = self.fusion.fuse(
            _make_utterance(),
            _make_acoustic(),
            _make_linguistic(urgency_score=0.5),
        )
        assert len(result.acoustic_contributions) > 0
        assert len(result.linguistic_contributions) > 0

    def test_none_acoustic_produces_zero_acoustic_score(self):
        result = self.fusion.fuse(
            _make_utterance(), None, _make_linguistic(urgency_score=0.5)
        )
        assert result.acoustic_score == pytest.approx(0.0)

    def test_reset_clears_ema(self):
        for _ in range(5):
            self.fusion.fuse(
                _make_utterance(),
                _make_acoustic(rms_mean=0.9),
                _make_linguistic(urgency_score=1.0),
            )
        high = self.fusion._prev_smoothed
        self.fusion.reset()
        assert self.fusion._prev_smoothed == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Regression: Bug 1 — normal speech must stay Low (no false Mild)
# ---------------------------------------------------------------------------

class TestSigmoidBiasNormalSpeechStaysLow:
    """Verify the sigmoid bias keeps neutral acoustic input well inside Low.

    Before the fix: all-zero z-scores (no personal baseline) produced
    acoustic_score = sigmoid(0) = 0.5, pushing raw_final to 0.30 and
    smoothed (after several calm utterances with EMA_ALPHA_UP=0.55) above
    SEVERITY_LOW_MAX=0.35, causing a spurious Mild label.

    After the fix: acoustic_score ≈ sigmoid(-3) ≈ 0.047, raw_final ≈ 0.028,
    smoothed stays well below 0.35 regardless of how many utterances are fused.
    """

    def setup_method(self):
        self.bm = BaselineManager()
        self.fusion = ScoreFusion(self.bm)

    def test_no_baseline_neutral_acoustic_score_near_zero(self):
        """Neutral features with no personal baseline must give a low acoustic score."""
        acoustic = _make_acoustic(
            rms_mean=0.02, rms_max=0.05, pitch_range=15.0,
            pitch_variance=80.0, voiced_ratio=0.65,
        )
        result = self.fusion.fuse(_make_utterance("hello how are you"), acoustic, _make_linguistic())
        assert result.acoustic_score < 0.20, (
            f"acoustic_score={result.acoustic_score:.4f} is too high for normal speech "
            "with no personal baseline (sigmoid bias not working)"
        )

    def test_calm_repeated_speech_stays_low_severity(self):
        """Multiple consecutive calm utterances must NOT accumulate to Mild severity."""
        self.fusion.reset()
        acoustic = _make_acoustic(
            rms_mean=0.02, rms_max=0.06, pitch_range=18.0,
            pitch_variance=90.0, voiced_ratio=0.70,
        )
        linguistic = _make_linguistic(negative_sentiment=0.05)
        for _ in range(10):
            result = self.fusion.fuse(_make_utterance("fine thanks"), acoustic, linguistic)

        assert result.severity == "Low", (
            f"severity={result.severity!r}, smoothed={result.smoothed_score:.4f} — "
            "calm repeated speech must not accumulate to Mild"
        )
        assert result.smoothed_score < 0.35, (
            f"smoothed_score={result.smoothed_score:.4f} exceeded SEVERITY_LOW_MAX=0.35 "
            "during sustained calm speech"
        )

    def test_raw_fused_score_near_zero_without_baseline(self):
        """Raw fused score for typical calm speech should be below 0.10."""
        acoustic = _make_acoustic(
            rms_mean=0.015, rms_max=0.04, voiced_ratio=0.60,
        )
        result = self.fusion.fuse(_make_utterance(), acoustic, _make_linguistic())
        assert result.raw_final_score < 0.15, (
            f"raw_final_score={result.raw_final_score:.4f} — expected < 0.15 for calm speech"
        )


# ---------------------------------------------------------------------------
# Regression: Bug 2 — high-energy screaming must always be detected
# ---------------------------------------------------------------------------

class TestScreamingAlwaysDetectedAboveAbsoluteThreshold:
    """High-energy vocalisation must trigger Screaming even without a baseline.

    Before the fix: the scream gate relied primarily on z-score-derived
    contribution values.  When the rolling baseline had absorbed loud speech,
    z-scores were small and the acoustic_score (now biased) was also low,
    so no path through the gate fired.

    The absolute-energy path (`absolute_energy_high`) is now the primary
    backstop: rms_mean >= 0.18 AND rms_max >= 0.65 always triggers Screaming
    regardless of the rolling baseline state.
    """

    def setup_method(self):
        from behaviour_classifier import BehaviourClassifier
        from score_fusion import ScoreFusion
        self.bm = BaselineManager()
        self.fusion = ScoreFusion(self.bm)
        self.clf = BehaviourClassifier()

    def _classify_acoustic(self, **acoustic_kwargs):
        acoustic = _make_acoustic(**acoustic_kwargs)
        result = self.fusion.fuse(
            _make_utterance("aaaaah"), acoustic, _make_linguistic()
        )
        result.acoustic_features = acoustic
        return self.clf.classify(result).behaviours

    def test_very_loud_sustained_scream_detected(self):
        """rms_mean well above threshold + rms_max above peak threshold."""
        labels = self._classify_acoustic(
            rms_mean=0.35, rms_max=0.80, voiced_ratio=0.75, clipping_ratio=0.0,
        )
        assert "Screaming" in labels, (
            f"Expected Screaming, got {labels}. "
            "Absolute-energy path did not fire."
        )

    def test_clipped_audio_detected_as_screaming(self):
        """Heavy clipping + moderate RMS should fire via clipping_high path."""
        labels = self._classify_acoustic(
            rms_mean=0.20, rms_max=0.90, voiced_ratio=0.0, clipping_ratio=0.15,
        )
        assert "Screaming" in labels, (
            f"Expected Screaming for clipped audio, got {labels}."
        )

    def test_normal_speech_rms_does_not_trigger_screaming(self):
        """Normal conversational RMS must NOT trigger screaming via absolute path."""
        labels = self._classify_acoustic(
            rms_mean=0.03, rms_max=0.12, voiced_ratio=0.70, clipping_ratio=0.0,
        )
        assert "Screaming" not in labels, (
            f"False screaming detection on normal speech: {labels}"
        )

    def test_scream_detected_even_when_acoustic_score_is_low(self):
        """Absolute path fires even if the biased acoustic_score is below 0.65."""
        from behaviour_classifier import BehaviourClassifier
        # Directly build a FusedResult with low acoustic_score but loud absolute values
        acoustic = _make_acoustic(
            rms_mean=0.30, rms_max=0.75, voiced_ratio=0.60, clipping_ratio=0.0,
        )
        import time
        from event_models import CommittedLine, FusedResult, Utterance
        ts = time.time()
        result = FusedResult(
            acoustic_score=0.20,   # deliberately low (e.g. baseline not converged)
            linguistic_score=0.0,
            raw_final_score=0.12,
            smoothed_score=0.10,
            severity="Low",
            reliability=0.9,
            utterance=Utterance(
                lines=[CommittedLine(text="aah", timestamp=ts)],
                start_time=ts - 2, end_time=ts,
            ),
            acoustic_features=acoustic,
            linguistic_features=_make_linguistic(),
            acoustic_contributions={"energy_above_baseline": 0.05, "energy_burst": 0.02},
            linguistic_contributions={},
        )
        clf = BehaviourClassifier()
        labels = clf.classify(result).behaviours
        assert "Screaming" in labels, (
            f"Screaming not detected even with rms_mean=0.30, rms_max=0.75. "
            f"Got: {labels}. Acoustic score was deliberately set to 0.20 to simulate "
            "no-baseline / biased-sigmoid scenario."
        )

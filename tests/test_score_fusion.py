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
        # With all-zero features and no baseline, z-scores are 0 → sigmoid(0)=0.5
        assert 0.0 <= result.acoustic_score <= 1.0
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

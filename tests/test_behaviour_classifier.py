"""Tests for the multi-label behaviour classifier.

Key spec scenarios:
- Calm swearing → NO verbal aggression
- Shouted neutral sentence → Screaming/shouting but NOT verbal aggression
- Calm explicit threat → threat cue detected, acoustic is lower
- Explicit threat while shouting → Possible verbal aggression + High severity
- Calm repeated question → Repeated questioning, low/mild severity
- No agitation → 'No audio agitation detected'
"""
from __future__ import annotations

import time
import pytest

from baseline_manager import BaselineManager
from behaviour_classifier import BehaviourClassifier
from event_models import (
    AcousticFeatureWindow,
    CommittedLine,
    FusedResult,
    LinguisticFeatures,
    Utterance,
)
from score_fusion import ScoreFusion


def _make_utterance(text: str = "test") -> Utterance:
    ts = time.time()
    return Utterance(
        lines=[CommittedLine(text=text, timestamp=ts)],
        start_time=ts - 2.0,
        end_time=ts,
    )


def _make_result(
    acoustic_score: float = 0.3,
    linguistic_score: float = 0.2,
    smoothed_score: float = 0.2,
    linguistic: LinguisticFeatures | None = None,
    acoustic: AcousticFeatureWindow | None = None,
    acoustic_contributions: dict | None = None,
) -> FusedResult:
    if linguistic is None:
        linguistic = LinguisticFeatures()
    ts = time.time()
    result = FusedResult(
        acoustic_score=acoustic_score,
        linguistic_score=linguistic_score,
        raw_final_score=0.6 * acoustic_score + 0.4 * linguistic_score,
        smoothed_score=smoothed_score,
        severity="Low" if smoothed_score < 0.35 else "Mild" if smoothed_score < 0.6 else "Moderate" if smoothed_score < 0.8 else "High",
        reliability=0.9,
        utterance=_make_utterance(),
        linguistic_features=linguistic,
        acoustic_features=acoustic,
        acoustic_contributions=acoustic_contributions or {},
        linguistic_contributions={},
    )
    return result


class TestBehaviourClassifier:
    def setup_method(self):
        self.clf = BehaviourClassifier()

    def _classify(self, result: FusedResult) -> list[str]:
        return self.clf.classify(result).behaviours

    # ---- No agitation -------------------------------------------------

    def test_calm_produces_no_agitation_label(self):
        result = _make_result(acoustic_score=0.1, linguistic_score=0.05, smoothed_score=0.1)
        labels = self._classify(result)
        assert "No audio agitation detected" in labels

    # ---- Calm swearing → NO verbal aggression -------------------------

    def test_calm_swearing_no_verbal_aggression(self):
        linguistic = LinguisticFeatures(
            profanity_score=0.33,
            negative_sentiment=0.3,
            threat_score=0.0,
            imperative_score=0.2,
        )
        # Low acoustic score (calm voice)
        result = _make_result(
            acoustic_score=0.25,
            smoothed_score=0.25,
            linguistic=linguistic,
        )
        labels = self._classify(result)
        assert "Cursing / verbal aggression" not in labels

    # ---- Shouted neutral → Screaming but not verbal aggression --------

    def test_shouting_without_threat_not_verbal_aggression(self):
        acoustic = AcousticFeatureWindow(
            start_time=time.time() - 2,
            end_time=time.time(),
            rms_mean=0.5,
            rms_max=0.9,
            voiced_ratio=0.75,
        )
        linguistic = LinguisticFeatures(
            negative_sentiment=0.1,
            threat_score=0.0,
            profanity_score=0.0,
        )
        result = _make_result(
            acoustic_score=0.8,
            smoothed_score=0.7,
            linguistic=linguistic,
            acoustic=acoustic,
            acoustic_contributions={
                "energy_above_baseline": 0.25,   # high energy
                "energy_burst": 0.18,             # high burst
            },
        )
        result.linguistic_features = linguistic
        result.acoustic_features = acoustic
        labels = self._classify(result)
        assert "Cursing / verbal aggression" not in labels
        assert "Screaming" in labels

    # ---- Calm threat → threat detected, not necessarily verbal aggression

    def test_calm_explicit_threat(self):
        linguistic = LinguisticFeatures(
            threat_score=0.5,
            negative_sentiment=0.4,
            profanity_score=0.0,
        )
        result = _make_result(
            acoustic_score=0.30,   # calm voice
            smoothed_score=0.30,
            linguistic=linguistic,
        )
        labels = self._classify(result)
        # Acoustic is below verbal aggression threshold (0.65)
        assert "Cursing / verbal aggression" not in labels

    # ---- Shouting + explicit threat → verbal aggression ---------------

    def test_shouting_plus_threat_triggers_verbal_aggression(self):
        acoustic = AcousticFeatureWindow(
            start_time=time.time() - 2,
            end_time=time.time(),
            rms_mean=0.6,
            rms_max=0.95,
            voiced_ratio=0.8,
        )
        linguistic = LinguisticFeatures(
            threat_score=0.7,
            negative_sentiment=0.65,
            profanity_score=0.33,
        )
        result = _make_result(
            acoustic_score=0.80,
            smoothed_score=0.82,
            linguistic=linguistic,
            acoustic=acoustic,
            acoustic_contributions={
                "energy_above_baseline": 0.25,
                "energy_burst": 0.18,
            },
        )
        result.linguistic_features = linguistic
        result.acoustic_features = acoustic
        labels = self._classify(result)
        assert "Cursing / verbal aggression" in labels

    # ---- Repeated question → labelled --------------------------------

    def test_repeated_questioning_detected(self):
        linguistic = LinguisticFeatures(
            question_repetition_score=0.75,
        )
        result = _make_result(
            linguistic=linguistic,
            smoothed_score=0.40,
        )
        labels = self._classify(result)
        assert "Repetitive sentences or questions" in labels

    # ---- Repeated verbalization --------------------------------------

    def test_repetitive_verbalization_detected(self):
        linguistic = LinguisticFeatures(
            repetition_score=0.70,
        )
        result = _make_result(linguistic=linguistic, smoothed_score=0.45)
        labels = self._classify(result)
        assert "Repetitive sentences or questions" in labels

    # ---- Distressed verbalization ------------------------------------

    def test_distressed_verbalization(self):
        linguistic = LinguisticFeatures(urgency_score=0.75)
        result = _make_result(
            acoustic_score=0.60,
            smoothed_score=0.55,
            linguistic=linguistic,
        )
        labels = self._classify(result)
        assert "Unmapped audio behaviour" in labels

    # ---- Multi-label -------------------------------------------------

    def test_multi_label_simultaneous(self):
        """Repeated question + distress can coexist."""
        linguistic = LinguisticFeatures(
            question_repetition_score=0.80,
            urgency_score=0.70,
        )
        result = _make_result(
            acoustic_score=0.65,
            smoothed_score=0.65,
            linguistic=linguistic,
        )
        labels = self._classify(result)
        assert "Repetitive sentences or questions" in labels
        assert "Unmapped audio behaviour" in labels

    # ---- No physical labels ------------------------------------------

    def test_no_physical_labels_ever_emitted(self):
        physical = {
            "Pacing", "General restlessness", "Hitting", "Kicking",
            "Grabbing", "Trying to leave the room",
        }
        linguistic = LinguisticFeatures(
            threat_score=1.0, urgency_score=1.0,
            question_repetition_score=1.0, repetition_score=1.0,
        )
        acoustic = AcousticFeatureWindow(
            start_time=time.time() - 2, end_time=time.time(),
            rms_mean=0.9, rms_max=1.0, voiced_ratio=0.9,
        )
        result = _make_result(
            acoustic_score=0.95, smoothed_score=0.95,
            linguistic=linguistic, acoustic=acoustic,
            acoustic_contributions={"energy_above_baseline": 0.3, "energy_burst": 0.2},
        )
        result.linguistic_features = linguistic
        result.acoustic_features = acoustic
        labels = set(self._classify(result))
        assert len(labels & physical) == 0

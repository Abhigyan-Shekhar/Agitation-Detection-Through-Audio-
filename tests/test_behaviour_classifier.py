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
    utterance_text: str = "test",
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
        utterance=_make_utterance(utterance_text),
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

    def _scream_like_acoustic(self, **kwargs) -> AcousticFeatureWindow:
        now = time.time()
        defaults = dict(
            start_time=now - 2.0,
            end_time=now,
            rms_mean=0.23,
            rms_max=0.82,
            rms_slope=0.003,
            pitch_median=430.0,
            pitch_range=260.0,
            pitch_variance=9000.0,
            zcr_mean=0.12,
            spectral_centroid=3800.0,
            spectral_rolloff=6500.0,
            harmonic_to_noise_ratio=1.0,
            voiced_ratio=0.45,
            pause_ratio=0.55,
            clipping_ratio=0.0,
        )
        defaults.update(kwargs)
        return AcousticFeatureWindow(**defaults)

    # ---- No agitation -------------------------------------------------

    def test_calm_produces_no_agitation_label(self):
        result = _make_result(acoustic_score=0.1, linguistic_score=0.05, smoothed_score=0.1)
        labels = self._classify(result)
        assert "No audio agitation detected" in labels

    # ---- Calm swearing → NO verbal aggression -------------------------

    def test_calm_swearing_no_verbal_aggression(self):
        linguistic = LinguisticFeatures(
            profanity_score=0.25,
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

    def test_explicit_curse_word_triggers_cursing_without_repetition(self):
        linguistic = LinguisticFeatures(
            profanity_score=0.70,
            negative_sentiment=0.2,
            threat_score=0.0,
            imperative_score=0.0,
        )
        result = _make_result(
            acoustic_score=0.20,
            smoothed_score=0.20,
            linguistic=linguistic,
            utterance_text="What the fuck is going on?",
        )
        labels = self._classify(result)
        assert "No audio agitation detected" not in labels
        assert "Cursing / verbal aggression" in labels

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

    def test_transcript_yelling_terms_trigger_screaming(self):
        linguistic = LinguisticFeatures(yelling_score=0.65)
        result = _make_result(
            acoustic_score=0.10,
            smoothed_score=0.20,
            linguistic=linguistic,
            utterance_text="Stop yelling at me.",
        )
        labels = self._classify(result)
        assert "Screaming" in labels
        assert "No audio agitation detected" not in labels

    def test_absolute_loud_clipped_audio_triggers_screaming_without_baseline_score(self):
        acoustic = AcousticFeatureWindow(
            start_time=time.time() - 2,
            end_time=time.time(),
            rms_mean=0.42,
            rms_max=1.05,
            voiced_ratio=0.0,
            clipping_ratio=0.08,
        )
        result = _make_result(
            acoustic_score=0.10,
            smoothed_score=0.20,
            acoustic=acoustic,
            linguistic=LinguisticFeatures(),
            acoustic_contributions={},
            utterance_text="Ah",
        )
        labels = self._classify(result)
        assert "Screaming" in labels
        assert "No audio agitation detected" not in labels

    def test_normal_volume_audio_does_not_trigger_screaming_absolute_fallback(self):
        acoustic = AcousticFeatureWindow(
            start_time=time.time() - 2,
            end_time=time.time(),
            rms_mean=0.04,
            rms_max=0.16,
            voiced_ratio=0.8,
            clipping_ratio=0.0,
        )
        result = _make_result(
            acoustic_score=0.10,
            smoothed_score=0.20,
            acoustic=acoustic,
            linguistic=LinguisticFeatures(),
            acoustic_contributions={},
            utterance_text="Ah",
        )
        labels = self._classify(result)
        assert "Screaming" not in labels

    def test_obvious_acoustic_screaming_detected_without_transcript_keywords(self):
        result = _make_result(
            acoustic_score=0.30,
            smoothed_score=0.30,
            acoustic=self._scream_like_acoustic(),
            linguistic=LinguisticFeatures(),
            utterance_text="ah",
        )
        labels = self._classify(result)
        assert "Screaming" in labels
        assert "No audio agitation detected" not in labels

    def test_repeated_scream_bursts_detected_from_peak_and_pitch_variability(self):
        acoustic = self._scream_like_acoustic(
            rms_mean=0.14,
            rms_max=0.88,
            rms_slope=0.004,
            pitch_median=360.0,
            pitch_range=340.0,
            pitch_variance=14000.0,
            voiced_ratio=0.35,
        )
        result = _make_result(
            acoustic_score=0.40,
            smoothed_score=0.35,
            acoustic=acoustic,
            linguistic=LinguisticFeatures(),
        )
        assert "Screaming" in self._classify(result)

    def test_short_scream_event_detected_when_onset_and_peak_are_strong(self):
        now = time.time()
        acoustic = self._scream_like_acoustic(
            start_time=now - 0.18,
            end_time=now,
            rms_mean=0.12,
            rms_max=0.92,
            rms_slope=0.006,
            pitch_median=520.0,
            spectral_centroid=4300.0,
            spectral_rolloff=7200.0,
        )
        result = _make_result(acoustic_score=0.35, smoothed_score=0.25, acoustic=acoustic)
        assert "Screaming" in self._classify(result)

    def test_prolonged_screaming_detected_without_abrupt_onset(self):
        acoustic = self._scream_like_acoustic(
            start_time=time.time() - 3.0,
            end_time=time.time(),
            rms_mean=0.21,
            rms_max=0.70,
            rms_slope=0.0001,
            pitch_median=390.0,
            pitch_range=220.0,
        )
        result = _make_result(acoustic_score=0.35, smoothed_score=0.35, acoustic=acoustic)
        assert "Screaming" in self._classify(result)

    def test_loud_talking_does_not_trigger_screaming(self):
        acoustic = self._scream_like_acoustic(
            rms_mean=0.24,
            rms_max=0.74,
            rms_slope=0.0002,
            pitch_median=170.0,
            pitch_range=45.0,
            pitch_variance=250.0,
            zcr_mean=0.04,
            spectral_centroid=1700.0,
            spectral_rolloff=3100.0,
            voiced_ratio=0.92,
        )
        result = _make_result(acoustic_score=0.82, smoothed_score=0.75, acoustic=acoustic)
        assert "Screaming" not in self._classify(result)

    def test_laughter_does_not_trigger_screaming(self):
        acoustic = self._scream_like_acoustic(
            rms_mean=0.16,
            rms_max=0.72,
            rms_slope=0.001,
            pitch_median=210.0,
            pitch_range=80.0,
            pitch_variance=900.0,
            zcr_mean=0.11,
            spectral_centroid=2600.0,
            spectral_rolloff=5200.0,
            harmonic_to_noise_ratio=-8.0,
            voiced_ratio=0.30,
        )
        result = _make_result(acoustic_score=0.65, smoothed_score=0.55, acoustic=acoustic)
        assert "Screaming" not in self._classify(result)

    def test_crying_like_vocalization_does_not_trigger_screaming(self):
        acoustic = self._scream_like_acoustic(
            rms_mean=0.10,
            rms_max=0.38,
            rms_slope=0.0004,
            pitch_median=280.0,
            pitch_range=120.0,
            pitch_variance=1500.0,
            zcr_mean=0.04,
            spectral_centroid=1600.0,
            spectral_rolloff=3000.0,
            voiced_ratio=0.55,
        )
        result = _make_result(acoustic_score=0.50, smoothed_score=0.45, acoustic=acoustic)
        assert "Screaming" not in self._classify(result)

    def test_caregiver_shouting_style_low_pitch_loud_speech_is_not_screaming(self):
        acoustic = self._scream_like_acoustic(
            rms_mean=0.26,
            rms_max=0.82,
            rms_slope=0.0005,
            pitch_median=155.0,
            pitch_range=60.0,
            pitch_variance=500.0,
            zcr_mean=0.05,
            spectral_centroid=1900.0,
            spectral_rolloff=3600.0,
            voiced_ratio=0.90,
        )
        result = _make_result(acoustic_score=0.86, smoothed_score=0.80, acoustic=acoustic)
        assert "Screaming" not in self._classify(result)

    def test_television_noise_does_not_trigger_screaming(self):
        acoustic = self._scream_like_acoustic(
            rms_mean=0.19,
            rms_max=0.76,
            rms_slope=0.0015,
            pitch_median=0.0,
            pitch_range=0.0,
            pitch_variance=0.0,
            zcr_mean=0.13,
            spectral_centroid=3600.0,
            spectral_rolloff=7200.0,
            harmonic_to_noise_ratio=-12.0,
            voiced_ratio=0.05,
        )
        result = _make_result(acoustic_score=0.78, smoothed_score=0.70, acoustic=acoustic)
        assert "Screaming" not in self._classify(result)

    def test_background_environmental_sound_does_not_trigger_screaming(self):
        acoustic = self._scream_like_acoustic(
            rms_mean=0.12,
            rms_max=0.58,
            rms_slope=0.003,
            pitch_median=0.0,
            pitch_range=0.0,
            pitch_variance=0.0,
            zcr_mean=0.16,
            spectral_centroid=5000.0,
            spectral_rolloff=7800.0,
            harmonic_to_noise_ratio=-15.0,
            voiced_ratio=0.0,
        )
        result = _make_result(acoustic_score=0.70, smoothed_score=0.60, acoustic=acoustic)
        assert "Screaming" not in self._classify(result)

    def test_verbal_sexual_advance_triggers_canonical_label(self):
        linguistic = LinguisticFeatures(sexual_advance_score=0.85)
        result = _make_result(
            acoustic_score=0.10,
            smoothed_score=0.20,
            linguistic=linguistic,
            utterance_text="Come to bed with me.",
        )
        labels = self._classify(result)
        assert "Making verbal sexual advances" in labels
        assert "No audio agitation detected" not in labels

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

    def test_repeated_leave_me_alone_detected_as_request_not_no_agitation(self):
        linguistic = LinguisticFeatures(
            repetition_score=0.85,
            urgency_score=0.33,
            imperative_score=1.0,
            evidence={"repetition": {"rep": 0.85, "q_rep": 0.0, "req_rep": 0.85}},
        )
        result = _make_result(
            linguistic=linguistic,
            smoothed_score=0.20,
            utterance_text="Hello? Leave me alone. Leave me alone. Leave me alone.",
        )
        classified = self.clf.classify(result)
        assert "No audio agitation detected" not in classified.behaviours
        assert "Constant unwarranted requests for attention/help" in classified.behaviours
        assert classified.behaviour_events[0].canonical_label == "Constant unwarranted requests for attention/help"

    def test_repeated_save_me_detected_as_request_not_no_agitation(self):
        linguistic = LinguisticFeatures(
            repetition_score=0.94,
            urgency_score=0.67,
            evidence={"repetition": {"rep": 0.94, "q_rep": 0.0, "req_rep": 0.94}},
        )
        result = _make_result(
            linguistic=linguistic,
            smoothed_score=0.20,
            utterance_text="save me save me save me help help help",
        )
        classified = self.clf.classify(result)
        assert "No audio agitation detected" not in classified.behaviours
        assert "Constant unwarranted requests for attention/help" in classified.behaviours

    def test_duplicate_canonical_labels_are_deduplicated(self):
        linguistic = LinguisticFeatures(
            repetition_score=0.95,
            question_repetition_score=0.95,
        )
        result = _make_result(linguistic=linguistic, smoothed_score=0.40)
        labels = self._classify(result)
        assert labels.count("Repetitive sentences or questions") == 1

    # ---- Distressed verbalization ------------------------------------

    def test_distressed_verbalization(self):
        linguistic = LinguisticFeatures(urgency_score=0.75)
        result = _make_result(
            acoustic_score=0.60,
            smoothed_score=0.55,
            linguistic=linguistic,
        )
        labels = self._classify(result)
        assert "Distressed/urgent verbalization" in labels

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
        assert "Distressed/urgent verbalization" in labels

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

    def test_complaint_score_triggers_complaining_label(self):
        linguistic = LinguisticFeatures(
            complaint_score=0.75,
            evidence={
                "complaint": {
                    "complaint_score": 0.75,
                    "complaint_keywords": ["tired of"],
                    "complaint_patterns_matched": ["tired_of_this"],
                    "complaint_confidence": 0.75,
                }
            },
        )
        result = _make_result(
            linguistic=linguistic,
            smoothed_score=0.20,
            utterance_text="I'm tired of this.",
        )
        classified = self.clf.classify(result)
        assert "Complaining" in classified.behaviours
        assert "No audio agitation detected" not in classified.behaviours
        assert classified.behaviour_events[0].canonical_label == "Complaining"

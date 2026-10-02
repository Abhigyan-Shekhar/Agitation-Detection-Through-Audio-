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
        # Simulate consecutive windows for persistence gates like screaming
        import time
        labels = []
        original_ts = getattr(result.acoustic_features, "end_time", None) or time.time()
        for i in range(3):
            # Clone and stagger the time
            step_ts = original_ts + i * 1.0
            
            import copy
            step_result = copy.copy(result)
            
            if step_result.acoustic_features:
                step_acoustic = copy.copy(step_result.acoustic_features)
                step_acoustic.start_time = step_ts - 2.0
                step_acoustic.end_time = step_ts
                step_result.acoustic_features = step_acoustic
            
            labels = self.clf.classify(step_result).behaviours
        return labels

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

    def test_transcript_mention_of_yelling_does_not_trigger_screaming(self):
        linguistic = LinguisticFeatures(yelling_score=0.65)
        result = _make_result(
            acoustic_score=0.10,
            smoothed_score=0.20,
            linguistic=linguistic,
            utterance_text="Stop yelling at me.",
        )
        labels = self._classify(result)
        assert "Screaming" not in labels
        assert "No audio agitation detected" in labels

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

    def test_high_confidence_urgent_language_is_not_suppressed(self):
        linguistic = LinguisticFeatures(urgency_score=0.90, evidence={"transcript": {"confidence": 0.95}})
        result = _make_result(acoustic_score=0.20, smoothed_score=0.20, linguistic=linguistic)
        labels = self._classify(result)
        assert "Distressed/urgent verbalization" in labels

    def test_extreme_short_scream_bypasses_persistence(self):
        acoustic = AcousticFeatureWindow(start_time=time.time() - 0.5, end_time=time.time(), rms_mean=0.55, rms_max=0.95, voiced_ratio=0.9)
        result = _make_result(acoustic_score=0.98, smoothed_score=0.80, acoustic=acoustic,
                              acoustic_contributions={"energy_above_baseline": 0.30, "energy_burst": 0.20})
        assert "Screaming" in self.clf.classify(result).behaviours

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

    def test_negativism_triggers_when_threshold_is_exceeded(self):
        linguistic = LinguisticFeatures(
            negativism_score=0.80,
            evidence={
                "negativism": {
                    "negativism_score": 0.80,
                    "categories": ["refusal"],
                    "matched_phrases": ["i won't"],
                }
            },
        )
        result = _make_result(
            linguistic=linguistic,
            smoothed_score=0.20,
            utterance_text="I won't take my medicine.",
        )
        classified = self.clf.classify(result)
        assert "Negativism" in classified.behaviours
        assert "No audio agitation detected" not in classified.behaviours
        assert classified.behaviour_events[0].canonical_label == "Negativism"

    def test_strange_noise_triggers_cmai_strange_noise_event(self):
        linguistic = LinguisticFeatures(
            strange_noise_score=0.85,
            evidence={
                "strange_noise": {
                    "strange_noise_score": 0.85,
                    "matched_labels": ["moaning"],
                    "source_datasets": ["OpenSLR SLR99 / Deeply Nonverbal Vocalization Dataset"],
                }
            },
        )
        result = _make_result(
            linguistic=linguistic,
            smoothed_score=0.20,
            utterance_text="[moaning]",
        )
        classified = self.clf.classify(result)
        assert "Making strange noises" in classified.behaviours
        assert "No audio agitation detected" not in classified.behaviours
        assert classified.behaviour_events[0].canonical_label == "Making strange noises"
        assert classified.behaviour_events[0].cmai_category == "Verbally non-aggressive: strange noises"

    def test_raw_acoustic_moaning_triggers_cmai_strange_noise_event(self):
        acoustic = AcousticFeatureWindow(
            start_time=time.time() - 2,
            end_time=time.time(),
            rms_mean=0.08,
            rms_max=0.18,
            non_speech_vocalization_score=0.82,
            non_speech_vocalization_label="moaning",
            non_speech_vocalization_evidence="acoustic moaning heuristic",
        )
        result = _make_result(
            acoustic_score=0.10,
            smoothed_score=0.20,
            acoustic=acoustic,
            linguistic=LinguisticFeatures(),
            utterance_text="",
        )
        classified = self.clf.classify(result)
        assert "Making strange noises" in classified.behaviours
        assert "No audio agitation detected" not in classified.behaviours
        assert classified.behaviour_events[0].canonical_label == "Making strange noises"

"""Qualitative regression fixtures for the seven supported audio behaviours.

These are deterministic unit fixtures, not accuracy measurements or a
substitute for a labelled audio evaluation set.
"""
from __future__ import annotations

import time

import pytest

from behaviour_classifier import BehaviourClassifier
from event_models import AcousticFeatureWindow, CommittedLine, FusedResult, LinguisticFeatures, Utterance
from linguistic_features import LinguisticAnalyzer


def _utterance(text: str, ts: float) -> Utterance:
    return Utterance(
        lines=[CommittedLine(text=text, timestamp=ts, transcript_confidence=0.92)],
        start_time=ts - 1.0,
        end_time=ts,
    )


def _classify(text: str, analyzer: LinguisticAnalyzer, ts: float) -> list[str]:
    utterance = _utterance(text, ts)
    linguistic = analyzer.analyze(utterance)
    result = FusedResult(
        acoustic_score=0.08,
        linguistic_score=0.0,
        raw_final_score=0.0,
        smoothed_score=0.1,
        utterance=utterance,
        linguistic_features=linguistic,
    )
    return BehaviourClassifier().classify(result).behaviours


@pytest.mark.parametrize(("text", "expected"), [
    ("This hurts and nobody listens to me.", "Complaining"),
    ("I will not take my medicine. Leave me alone.", "Negativism"),
])
def test_single_utterance_linguistic_behaviours(text: str, expected: str):
    assert expected in _classify(text, LinguisticAnalyzer(), time.time())


def test_repeated_question_and_request_use_transcript_history():
    analyzer = LinguisticAnalyzer()
    ts = time.time()
    analyzer.analyze(_utterance("Can I go home?", ts))
    second = analyzer.analyze(_utterance("Can I go home please?", ts + 4))
    assert second.question_repetition_score >= 0.70

    analyzer = LinguisticAnalyzer()
    analyzer.analyze(_utterance("Please help me get home.", ts))
    second = analyzer.analyze(_utterance("Please help me get home now.", ts + 4))
    assert second.evidence["repetition"]["req_rep"] >= 0.65


def test_scream_requires_three_distinct_acoustic_windows():
    classifier = BehaviourClassifier()
    ts = time.time()
    labels: list[str] = []
    for offset in (0.0, 0.5, 1.5):
        acoustic = AcousticFeatureWindow(
            start_time=ts + offset - 2,
            end_time=ts + offset,
            rms_mean=0.35,
            rms_max=0.82,
            voiced_ratio=0.8,
        )
        result = FusedResult(
            acoustic_score=0.9, smoothed_score=0.8,
            acoustic_features=acoustic,
            linguistic_features=LinguisticFeatures(),
            acoustic_contributions={"energy_above_baseline": 0.3, "energy_burst": 0.2},
        )
        labels = classifier.classify(result).behaviours
    assert "Screaming" in labels


def test_calm_textual_yelling_and_profanity_discussion_are_not_events():
    assert "Screaming" not in _classify("Please stop yelling at me.", LinguisticAnalyzer(), time.time())
    assert "Cursing / verbal aggression" not in _classify("Fuck my life.", LinguisticAnalyzer(), time.time())
    assert "Cursing / verbal aggression" not in _classify("Shit, I forgot my keys.", LinguisticAnalyzer(), time.time())
    assert "Cursing / verbal aggression" not in _classify(
        "The word fuck is a profanity in this example.", LinguisticAnalyzer(), time.time()
    )

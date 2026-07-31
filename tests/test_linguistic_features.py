"""Tests for linguistic feature extraction.

Key test scenarios from the spec:
- Calm casual swearing → no verbal aggression flag
- Explicit threat → threat score high
- Repeated question → q_rep score high
- Hinglish urgency → urgency score positive
- Neutral repeated phrase → repetition score high
"""
from __future__ import annotations

import time
import pytest

from linguistic_features import LinguisticAnalyzer, _normalize, _is_question, _content_words
from event_models import Utterance, CommittedLine


def _make_utterance(text: str) -> Utterance:
    ts = time.time()
    line = CommittedLine(text=text, timestamp=ts)
    return Utterance(lines=[line], start_time=ts - 2.0, end_time=ts)


class TestNormalization:
    def test_lowercase(self):
        assert _normalize("HELLO World") == "hello world"

    def test_strips_punctuation(self):
        result = _normalize("Hello, world!")
        assert "hello" in result
        assert "world" in result

    def test_normalises_whitespace(self):
        result = _normalize("  hello   world  ")
        assert "  " not in result


class TestIsQuestion:
    def test_ends_with_question_mark(self):
        assert _is_question("Where is my home?") is True

    def test_starts_with_why(self):
        assert _is_question("why can't I go home") is True

    def test_starts_with_hinglish_kyun(self):
        assert _is_question("kyun nahi jaate") is True

    def test_statement_not_question(self):
        assert _is_question("I want to go home.") is False


class TestLinguisticAnalyzer:
    def setup_method(self):
        self.analyzer = LinguisticAnalyzer()

    # ---- Repetition ---------------------------------------------------

    def test_no_history_zero_repetition(self):
        u = _make_utterance("I want to go home.")
        feats = self.analyzer.analyze(u)
        assert feats.repetition_score == pytest.approx(0.0, abs=0.01)

    def test_repeated_phrase_scores_high(self):
        for phrase in [
            "I want to go home.",
            "Can I go home?",
            "Please take me home.",
        ]:
            self.analyzer.analyze(_make_utterance(phrase))
        # After building history, same theme should score high
        feats = self.analyzer.analyze(_make_utterance("Why can't I go home?"))
        assert feats.repetition_score > 0.40

    def test_repetition_within_single_utterance_scores_high(self):
        feats = self.analyzer.analyze(
            _make_utterance("Leave me alone. Leave me alone. Leave me alone.")
        )
        assert feats.repetition_score >= 0.65
        assert feats.evidence["repetition"]["req_rep"] >= 0.65

    def test_repetition_within_single_utterance_without_punctuation_scores_high(self):
        feats = self.analyzer.analyze(
            _make_utterance("leave me alone leave me alone leave me alone")
        )
        assert feats.repetition_score >= 0.65

    def test_repeated_save_me_counts_as_request_repetition(self):
        feats = self.analyzer.analyze(
            _make_utterance("save me save me save me help help help")
        )
        assert feats.repetition_score >= 0.65
        assert feats.evidence["repetition"]["req_rep"] >= 0.65
        assert feats.urgency_score > 0.30

    def test_mic_check_question_is_not_request_repetition(self):
        feats = self.analyzer.analyze(
            _make_utterance("Can you hear me? Can you hear me?")
        )
        assert feats.evidence["repetition"]["req_rep"] == 0.0

    def test_question_repetition(self):
        self.analyzer.analyze(_make_utterance("Why can't I go home?"))
        self.analyzer.analyze(_make_utterance("Where is my home?"))
        feats = self.analyzer.analyze(_make_utterance("Why am I not going home?"))
        assert feats.question_repetition_score > 0.30

    # ---- Sentiment / Urgency / Threat ---------------------------------

    def test_calm_swear_low_threat(self):
        """Casual swearing should NOT trigger high threat score."""
        feats = self.analyzer.analyze(_make_utterance("Oh damn, I forgot."))
        assert feats.threat_score < 0.10

    def test_explicit_threat_scores_high(self):
        feats = self.analyzer.analyze(_make_utterance("I will hit you if you don't stop."))
        assert feats.threat_score > 0.40

    def test_escape_urgency(self):
        feats = self.analyzer.analyze(_make_utterance("Please let me go home, I need to leave now."))
        assert feats.urgency_score > 0.20

    def test_hinglish_urgency(self):
        feats = self.analyzer.analyze(_make_utterance("Mujhe jaana hai, abhi jaana hai."))
        assert feats.urgency_score > 0.10

    def test_negative_sentiment_on_sad_text(self):
        feats = self.analyzer.analyze(_make_utterance("This is terrible, I hate this place."))
        assert feats.negative_sentiment > 0.10

    def test_neutral_text_low_scores(self):
        feats = self.analyzer.analyze(_make_utterance("The weather is nice today."))
        assert feats.urgency_score < 0.10
        assert feats.threat_score == pytest.approx(0.0, abs=0.01)

    # ---- Profanity ----------------------------------------------------

    def test_profanity_detected(self):
        feats = self.analyzer.analyze(_make_utterance("What the fuck is going on?"))
        assert feats.profanity_score >= 0.50

    def test_common_english_curse_words_detected(self):
        for phrase in [
            "You are an asshole.",
            "This is bullshit.",
            "Stop being a bitch.",
            "What a dick.",
        ]:
            feats = self.analyzer.analyze(_make_utterance(phrase))
            assert feats.profanity_score >= 0.50

    def test_profanity_score_bounded(self):
        feats = self.analyzer.analyze(_make_utterance("fuck shit damn bastard"))
        assert feats.profanity_score <= 1.0

    def test_mild_damn_is_low_strength_profanity(self):
        feats = self.analyzer.analyze(_make_utterance("Oh damn, I forgot."))
        assert 0.0 < feats.profanity_score < 0.50

    # ---- Yelling ------------------------------------------------------

    def test_yelling_terms_detected(self):
        feats = self.analyzer.analyze(_make_utterance("Stop yelling at me."))
        assert feats.yelling_score >= 0.50

    def test_exclamation_and_caps_yelling_cues_detected(self):
        feats = self.analyzer.analyze(_make_utterance("STOP!! LISTEN!!"))
        assert feats.yelling_score >= 0.50

    # ---- Empty / edge cases ------------------------------------------

    def test_empty_utterance_returns_zero_scores(self):
        u = _make_utterance("")
        feats = self.analyzer.analyze(u)
        assert feats.repetition_score == 0.0
        assert feats.urgency_score == 0.0
        assert feats.threat_score == 0.0

    def test_all_scores_bounded(self):
        u = _make_utterance("I will kill you! Help! Bachao! Jaldi!")
        feats = self.analyzer.analyze(u)
        for attr in [
            "repetition_score", "question_repetition_score",
            "negative_sentiment", "urgency_score", "threat_score",
            "profanity_score", "imperative_score", "yelling_score",
        ]:
            val = getattr(feats, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of bounds"

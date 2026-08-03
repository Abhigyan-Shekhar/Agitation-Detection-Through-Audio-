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

    # ---- Verbal sexual advances -------------------------------------

    def test_verbal_sexual_advance_proposition_detected(self):
        feats = self.analyzer.analyze(_make_utterance("Come to bed with me."))
        assert feats.sexual_advance_score >= 0.60

    def test_verbal_sexualized_comment_detected(self):
        feats = self.analyzer.analyze(_make_utterance("You look so sexy."))
        assert feats.sexual_advance_score >= 0.60

    def test_clinical_verbal_sexual_advance_phrase_not_patient_utterance(self):
        feats = self.analyzer.analyze(
            _make_utterance("The DAVE dataset includes verbal sexual advances.")
        )
        assert feats.sexual_advance_score == 0.0

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
            "sexual_advance_score", "complaint_score", "strange_noise_score",
        ]:
            val = getattr(feats, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of bounds"


class TestNegativismDetection:
    def setup_method(self):
        self.analyzer = LinguisticAnalyzer()

    @pytest.mark.parametrize("phrase", [
        "I won't take my medicine.",
        "I will not do that.",
        "I'm not doing that.",
        "I refuse.",
        "No.",
    ])
    def test_negativism_refusal_examples(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.negativism_score >= 0.55
        assert feats.evidence["negativism"]["categories"]

    @pytest.mark.parametrize("phrase", [
        "Don't touch me.",
        "Stop bothering me.",
        "Leave me alone.",
        "Go away.",
    ])
    def test_negativism_resistance_examples(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.negativism_score >= 0.55

    @pytest.mark.parametrize("phrase", [
        "I'm not taking my medicine.",
        "I'm not going.",
        "I'm staying here.",
        "I don't want to.",
    ])
    def test_negativism_non_compliance_examples(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.negativism_score >= 0.55

    @pytest.mark.parametrize("phrase", [
        "You can't make me.",
        "Don't tell me what to do.",
        "I decide.",
        "I said no.",
    ])
    def test_negativism_defiance_examples(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.negativism_score >= 0.55

    def test_negativism_detects_mixed_sentences_with_refusal(self):
        feats = self.analyzer.analyze(_make_utterance("I'm sad but I won't take my medicine."))
        assert feats.negativism_score >= 0.55

    @pytest.mark.parametrize("phrase", [
        "I won't.",
        "I wont.",
        "I will not.",
        "I'm not.",
        "I am not.",
        "Don't.",
        "Do not.",
    ])
    def test_negativism_handles_contractions_and_variants(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.negativism_score >= 0.55

    def test_repeated_refusals_raise_confidence(self):
        feats = self.analyzer.analyze(_make_utterance("I won't. I won't. I won't."))
        assert feats.negativism_score >= 0.80

    @pytest.mark.parametrize("phrase", [
        "The CMAI includes Negativism.",
        "The dataset contains Negativism labels.",
    ])
    def test_documentation_text_does_not_trigger_negativism(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.negativism_score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("phrase", [
        "I'm sad.",
        "I miss my daughter.",
        "Today is terrible.",
        "Life is hard.",
        "I'm depressed.",
        "Everything is bad.",
        "I feel awful.",
    ])
    def test_emotional_state_sentences_do_not_trigger_negativism(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.negativism_score == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("phrase", [
        "I'm tired today.",
        "I have pain.",
        "I feel lonely.",
    ])
    def test_non_oppositional_states_do_not_trigger_negativism(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.negativism_score == pytest.approx(0.0, abs=0.01)


class TestComplaintDetection:
    def setup_method(self):
        self.analyzer = LinguisticAnalyzer()

    @pytest.mark.parametrize("phrase", [
        "I'm tired of this.",
        "Nobody listens to me.",
        "Everything is wrong.",
        "This hurts.",
        "I don't like this.",
    ])
    def test_complaint_positive_examples(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.complaint_score >= 0.55
        assert feats.evidence["complaint"]["complaint_patterns_matched"]
        assert feats.evidence["complaint"]["complaint_confidence"] >= 0.55

    @pytest.mark.parametrize("phrase", [
        "Hello.",
        "Thank you.",
        "Good morning.",
        "It is raining.",
        "My name is John.",
        "I am sad.",
        "I'm worried.",
        "I want to go home.",
    ])
    def test_complaint_negative_and_boundary_examples(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.complaint_score == pytest.approx(0.0, abs=0.01)


class TestStrangeNoiseDetection:
    def setup_method(self):
        self.analyzer = LinguisticAnalyzer()

    @pytest.mark.parametrize("phrase,expected_label", [
        ("[moaning]", "moaning"),
        ("Patient is groaning and sighing.", "groaning"),
        ("[throat clearing]", "throat clearing"),
        ("teeth chattering", "teeth chattering"),
        ("lip smacking", "lip smacking"),
        ("non-speech human vocalization", "non-speech human vocalization"),
    ])
    def test_dataset_labels_score_as_strange_noise(self, phrase, expected_label):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.strange_noise_score >= 0.60
        assert expected_label in feats.evidence["strange_noise"]["matched_labels"]
        assert feats.evidence["strange_noise"]["source_datasets"]

    @pytest.mark.parametrize("phrase", [
        "The OpenSLR dataset includes coughing and laughing labels.",
        "The CMAI category is making strange noises.",
        "The dataset contains throat clearing as a class.",
    ])
    def test_documentation_text_does_not_trigger_strange_noise(self, phrase):
        feats = self.analyzer.analyze(_make_utterance(phrase))
        assert feats.strange_noise_score == pytest.approx(0.0, abs=0.01)

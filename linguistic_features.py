"""Linguistic feature extraction for completed utterances.

Operates on the committed text from local transcriber. Maintains a rolling
30–60 second transcript history and computes:

A. repetition_score       — word overlap + n-gram + fuzzy similarity
B. question_repetition    — interrogative sentence matching
C. negative_sentiment     — VADER neg score + Hinglish lexicon
D. urgency_score          — weighted keyword matching
E. threat_score           — regex pattern matching
F. profanity_score        — raw count, intentionally weak weight
G. imperative_score       — command patterns (modifier for verbal-aggr rule)

Design notes
------------
* Stop-word filtering is applied before word repetition to avoid
  penalising normal filler like "I", "the", "a".
* VADER is presented as an English baseline only. Hinglish coverage is
  handled by a supplementary lexicon.
* Threat detection uses regex patterns — do NOT claim clinical accuracy.
* Speech rate is intentionally EXCLUDED here (it lives in acoustic branch
  to avoid double-counting in the final fusion).
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from typing import Deque

import numpy as np

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _vader = SentimentIntensityAnalyzer()
    _VADER_AVAILABLE = True
except ImportError:
    _vader = None
    _VADER_AVAILABLE = False

try:
    from rapidfuzz import fuzz  # type: ignore
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

import config
from event_models import LinguisticFeatures, Utterance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop words (excluded from repetition scoring)
# ---------------------------------------------------------------------------

STOP_WORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "shall", "may", "might", "must", "can", "and", "or",
    "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "into", "then", "than", "so", "if", "not", "no", "nor", "just",
    "that", "this", "these", "those", "what", "which", "who", "whom", "how",
    "when", "where", "why", "all", "each", "every",
    # Common Hinglish fillers
    "hai", "hain", "tha", "thi", "mein", "ko", "se", "aur", "bhi", "toh",
    "jo", "ki", "ka", "ke", "ho", "kya",
})

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

HELP_TERMS: frozenset[str] = frozenset({
    "help", "bachao", "doctor", "nurse", "ambulance", "emergency",
    "please help", "somebody help",
})

IMMEDIACY_TERMS: frozenset[str] = frozenset({
    "now", "immediately", "abhi", "jaldi", "right now", "hurry", "quick",
    "quickly", "at once", "asap",
})

STOP_TERMS: frozenset[str] = frozenset({
    "stop", "leave me", "leave me alone", "don't", "mat karo", "chhodo",
    "band karo", "ruko", "please stop", "get away", "go away",
})

ESCAPE_TERMS: frozenset[str] = frozenset({
    "let me go", "take me home", "mujhe jaana hai", "i want to go home",
    "i need to go home", "ghar jaana hai", "i want to leave",
    "please let me go",
})

REQUEST_TERMS: frozenset[str] = frozenset({
    "please", "can you", "could you", "will you", "would you",
    "i need", "i want", "give me", "bring me", "take me", "mujhe chahiye",
    "mujhe do",
})

INTERROGATIVE_PREFIXES: tuple[str, ...] = (
    "why", "where", "when", "who", "what", "how", "can i", "will you",
    "could you", "would you", "is it", "are you", "do you",
    # Hinglish
    "kyun", "kahan", "kab", "kaun", "kaise",
)

THREAT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bi(?:'ll| will| am going to) (?:hit|hurt|kill|break|smash|attack|punch)\b", re.I),
    re.compile(r"\bget away\b.*\bor else\b", re.I),
    re.compile(r"\bor(?:\s+else)?\s+i(?:'ll| will)\b", re.I),
    re.compile(r"\bmaar\s+(?:dunga|dungi|deta|deti)\b", re.I),
    re.compile(r"\btumhe\s+(?:maar|dhunga|nahi chhodunga)\b", re.I),
    re.compile(r"\bi(?:'ll| will) (?:destroy|ruin|throw|smash)\b", re.I),
]

# ---------------------------------------------------------------------------
# Hinglish negative lexicon (supplement for VADER)
# ---------------------------------------------------------------------------

NEGATIVE_HINGLISH: frozenset[str] = frozenset({
    "nahi", "mat", "chhodo", "jao", "gussa", "dard", "pareshan",
    "bekaar", "band karo", "rona", "takleef", "mushkil", "bura",
    "galat", "bekar", "dukh", "taklif", "khafa", "chinta",
})

PROFANITY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bf+u+c+k+\b", re.I),
    re.compile(r"\bsh+i+t+\b", re.I),
    re.compile(r"\bd+a+m+n+\b", re.I),
    re.compile(r"\bb+a+s+t+a+r+d+\b", re.I),
    re.compile(r"\bsaala\b", re.I),
    re.compile(r"\bkamina\b", re.I),
    re.compile(r"\bharamzada\b", re.I),
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, normalise whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_words(text: str) -> list[str]:
    return [w for w in _normalize(text).split() if w not in STOP_WORDS and len(w) > 1]


def _ngrams(words: list[str], n: int) -> set[str]:
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _is_question(text: str) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    lower = _normalize(t)
    return any(lower.startswith(p) for p in INTERROGATIVE_PREFIXES)


def _is_request(text: str) -> bool:
    lower = _normalize(text)
    return any(term in lower for term in REQUEST_TERMS)


def _keyword_score(text: str, terms: frozenset[str]) -> float:
    lower = _normalize(text)
    matches = sum(1 for term in terms if term in lower)
    return min(1.0, matches / max(len(terms) * 0.05, 1))


def _fuzzy_similarity(a: str, b: str) -> float:
    if not _RAPIDFUZZ_AVAILABLE:
        # Fallback: Jaccard over content words
        wa, wb = set(_content_words(a)), set(_content_words(b))
        if not wa and not wb:
            return 1.0
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)
    return fuzz.token_sort_ratio(a, b) / 100.0


# ---------------------------------------------------------------------------
# History record
# ---------------------------------------------------------------------------

class _TranscriptRecord:
    __slots__ = ("text", "timestamp", "is_question", "is_request")

    def __init__(self, text: str, timestamp: float) -> None:
        self.text = text
        self.timestamp = timestamp
        self.is_question = _is_question(text)
        self.is_request = _is_request(text)


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class LinguisticAnalyzer:
    """Stateful analyser that computes linguistic features per utterance.

    Maintains a rolling history of the last ``history_sec`` seconds of
    committed text for repetition and question-tracking.

    Parameters
    ----------
    history_sec:
        How far back to look for repetition / question matching.
    """

    def __init__(self, history_sec: float = config.TRANSCRIPT_HISTORY_SEC) -> None:
        self._history_sec = history_sec
        max_records = int(history_sec / 2) + 10  # generous upper bound
        self._history: Deque[_TranscriptRecord] = deque(maxlen=max_records)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, utterance: Utterance) -> LinguisticFeatures:
        """Compute all linguistic features for a completed utterance.

        Updates the rolling history after analysis.
        """
        text = utterance.full_text
        logger.info("BEHAVIOUR_TRACE linguistic_input transcript=%r", text)
        if not text.strip():
            logger.info("BEHAVIOUR_TRACE linguistic_output empty_transcript=True")
            return LinguisticFeatures()

        self._prune_history(utterance.end_time)

        evidence: dict = {}

        # A. Repetition
        rep, q_rep, req_rep = self._repetition_scores(text)
        evidence["repetition"] = {"rep": rep, "q_rep": q_rep, "req_rep": req_rep}

        # B. Sentiment
        neg_sentiment = self._sentiment_score(text)
        evidence["neg_sentiment"] = neg_sentiment

        # C. Urgency
        urgency = self._urgency_score(text)
        evidence["urgency"] = urgency

        # D. Threat
        threat = self._threat_score(text)
        evidence["threat"] = threat

        # E. Profanity
        profanity = self._profanity_score(text)
        evidence["profanity"] = profanity

        # F. Imperative
        imperative = self._imperative_score(text)
        evidence["imperative"] = imperative

        # Update history
        self._history.append(_TranscriptRecord(text, utterance.end_time))

        features = LinguisticFeatures(
            repetition_score=float(np.clip(rep, 0.0, 1.0)),
            question_repetition_score=float(np.clip(q_rep, 0.0, 1.0)),
            negative_sentiment=float(np.clip(neg_sentiment, 0.0, 1.0)),
            urgency_score=float(np.clip(urgency, 0.0, 1.0)),
            threat_score=float(np.clip(threat, 0.0, 1.0)),
            profanity_score=float(np.clip(profanity, 0.0, 1.0)),
            imperative_score=float(np.clip(imperative, 0.0, 1.0)),
            evidence=evidence,
        )
        logger.info(
            "BEHAVIOUR_TRACE linguistic_output repetition=%.3f question_repetition=%.3f negative=%.3f urgency=%.3f threat=%.3f profanity=%.3f imperative=%.3f",
            features.repetition_score,
            features.question_repetition_score,
            features.negative_sentiment,
            features.urgency_score,
            features.threat_score,
            features.profanity_score,
            features.imperative_score,
        )
        return features

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _repetition_scores(self, text: str) -> tuple[float, float, float]:
        """Return (repetition_score, question_repetition_score, request_repetition_score)."""
        if not self._history:
            return 0.0, 0.0, 0.0

        current_words = _content_words(text)
        current_3grams = _ngrams(current_words, 3)
        is_q = _is_question(text)
        is_req = _is_request(text)

        word_sims, phrase_sims, fuzzy_sims = [], [], []
        q_sims, req_sims = [], []

        for rec in self._history:
            hist_words = _content_words(rec.text)
            hist_3grams = _ngrams(hist_words, 3)

            # Word overlap (Jaccard)
            if current_words and hist_words:
                cw_set = set(current_words)
                hw_set = set(hist_words)
                word_sim = len(cw_set & hw_set) / len(cw_set | hw_set)
                word_sims.append(word_sim)

            # N-gram phrase overlap
            if current_3grams and hist_3grams:
                phrase_sim = len(current_3grams & hist_3grams) / len(current_3grams | hist_3grams)
                phrase_sims.append(phrase_sim)

            # Fuzzy full-sentence similarity
            fuzzy_sims.append(_fuzzy_similarity(text, rec.text))

            # Question repetition
            if is_q and rec.is_question:
                q_sims.append(_fuzzy_similarity(text, rec.text))

            # Request repetition
            if is_req and rec.is_request:
                req_sims.append(_fuzzy_similarity(text, rec.text))

        def _agg(sims: list[float]) -> float:
            return float(np.max(sims)) if sims else 0.0

        rep = (
            0.25 * _agg(word_sims)
            + 0.35 * _agg(phrase_sims)
            + 0.40 * _agg(fuzzy_sims)
        )
        q_rep = _agg(q_sims)
        req_rep = _agg(req_sims)

        return rep, q_rep, req_rep

    def _sentiment_score(self, text: str) -> float:
        score = 0.0

        # VADER (English)
        if _VADER_AVAILABLE and _vader is not None:
            vader_scores = _vader.polarity_scores(text)
            score = max(score, vader_scores["neg"])

        # Hinglish supplement
        words = set(_normalize(text).split())
        hinglish_matches = len(words & NEGATIVE_HINGLISH)
        hinglish_score = min(1.0, hinglish_matches / 3.0)
        score = max(score, hinglish_score)

        return score

    def _urgency_score(self, text: str) -> float:
        lower = _normalize(text)
        matches = 0.0
        # Weighted contribution by term category
        for term in HELP_TERMS:
            if term in lower:
                matches += 2.0          # help terms are strongest urgency signals
        for term in IMMEDIACY_TERMS:
            if term in lower:
                matches += 1.5
        for term in STOP_TERMS:
            if term in lower:
                matches += 1.0
        for term in ESCAPE_TERMS:
            if term in lower:
                matches += 2.0
        # Normalise: 3 weighted matches → score ≈ 1.0
        return min(1.0, matches / 6.0)

    def _threat_score(self, text: str) -> float:
        matches = sum(1 for p in THREAT_PATTERNS if p.search(text))
        return min(1.0, matches / 2.0)

    def _profanity_score(self, text: str) -> float:
        matches = sum(1 for p in PROFANITY_PATTERNS if p.search(text))
        return min(1.0, matches / 3.0)

    def _imperative_score(self, text: str) -> float:
        """Simple heuristic: imperatives tend to start with a base verb."""
        lower = text.strip().lower()
        imperative_verbs = (
            "stop", "leave", "go", "come", "take", "give", "bring",
            "sit", "stand", "help", "let", "don't", "do not",
            "jao", "ruko", "aao", "do", "mat",
        )
        words = lower.split()
        if words and words[0] in imperative_verbs:
            return 1.0
        matches = sum(1 for v in imperative_verbs if v in lower)
        return min(1.0, matches / 3.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prune_history(self, current_time: float) -> None:
        cutoff = current_time - self._history_sec
        while self._history and self._history[0].timestamp < cutoff:
            self._history.popleft()

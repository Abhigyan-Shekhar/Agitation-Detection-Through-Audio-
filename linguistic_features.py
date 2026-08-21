"""Linguistic feature extraction for completed utterances.

Operates on the committed text from local transcriber. Maintains a rolling
30–60 second transcript history and computes:

A. repetition_score       — word overlap + n-gram + fuzzy similarity
B. question_repetition    — interrogative sentence matching
C. negative_sentiment     — VADER neg score + Hinglish lexicon
D. urgency_score          — weighted keyword matching
E. threat_score           — regex pattern matching
F. profanity_score        — weighted common profanity matches
G. imperative_score       — command patterns (modifier for verbal-aggr rule)
H. yelling_score          — transcript yelling/shouting cues
I. sexual_advance_score   — sexualized verbal propositions/comments
J. complaint_score         — dissatisfaction/discomfort complaint semantics
K. negativism_score       — clinically-oriented refusal/resistance/non-compliance/defiance heuristic
L. strange_noise_score    — dataset-label cues for non-speech human vocalizations

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
from strange_noise_labels import STRANGE_NOISE_LABELS

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
    "please help", "somebody help", "save me", "help me",
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
    "please help", "help me", "save me", "somebody help",
    "could you help", "will you help", "would you help",
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
# Negative lexicon (supplement for VADER)
# ---------------------------------------------------------------------------

NEGATIVE_HINGLISH: frozenset[str] = frozenset({
    "nahi", "mat", "chhodo", "jao", "gussa", "dard", "pareshan",
    "bekaar", "band karo", "rona", "takleef", "mushkil", "bura",
    "galat", "bekar", "dukh", "taklif", "khafa", "chinta",
    "terrible", "hate", "angry", "upset", "bad", "awful", "scared",
    "afraid", "hurt", "pain", "leave", "alone",
})

PROFANITY_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\bmother\s*f+u+c+k+(?:er|ing)?\b", re.I), 0.9),
    (re.compile(r"\bf+u+c+k+(?:er|ing|ed)?\b", re.I), 0.7),
    (re.compile(r"\bsh+i+t+(?:ty|head)?\b", re.I), 0.55),
    (re.compile(r"\bb+u+l+l+s+h+i+t+\b", re.I), 0.6),
    (re.compile(r"\ba+s+s+h+o+l+e+s?\b", re.I), 0.7),
    (re.compile(r"\bb+i+t+c+h+(?:es|ing)?\b", re.I), 0.65),
    (re.compile(r"\bb+a+s+t+a+r+d+s?\b", re.I), 0.55),
    (re.compile(r"\bd+i+c+k+s?\b", re.I), 0.55),
    (re.compile(r"\bp+r+i+c+k+s?\b", re.I), 0.55),
    (re.compile(r"\bc+u+n+t+s?\b", re.I), 0.75),
    (re.compile(r"\bson\s+of\s+a\s+b+i+t+c+h+\b", re.I), 0.8),
    (re.compile(r"\bd+a+m+n+(?:ed|it)?\b", re.I), 0.25),
    (re.compile(r"\bcr+a+p+\b", re.I), 0.2),
    (re.compile(r"\bsaala\b", re.I), 0.45),
    (re.compile(r"\bkamina\b", re.I), 0.45),
    (re.compile(r"\bharamzada\b", re.I), 0.55),
)

# An ASR transcript may quote or discuss a profanity without containing a
# profanity event.  These contexts are deliberately narrow so genuine speech
# such as "that was fucking painful" remains detectable.
_PROFANITY_META_CONTEXT = re.compile(
    r"\b(?:the\s+(?:word|term)|a\s+(?:curse|swear)\s+word|profanity|"
    r"profane\s+(?:word|language)|how\s+to\s+spell|do\s+not\s+say)\b",
    re.I,
)

YELLING_TERMS: frozenset[str] = frozenset({
    "shout", "shouting", "yell", "yelling", "scream", "screaming",
    "shut up", "shut the hell up", "stop shouting", "stop yelling",
    "stop screaming", "loud voice",
})

LOUD_INTERJECTIONS: frozenset[str] = frozenset({
    "hey", "stop", "no", "help", "leave", "listen",
})

ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")
EXCLAMATION_RE = re.compile(r"!{1,}")

SEXUAL_ADVANCE_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\b(?:have sex|sleep with me|go to bed with me|come to bed|make love)\b", re.I), 0.95),
    (re.compile(r"\b(?:kiss me|let me kiss you|give me a kiss)\b", re.I), 0.65),
    (re.compile(r"\b(?:touch me|let me touch you|can i touch you)\b", re.I), 0.70),
    (re.compile(r"\b(?:take off|remove|open) (?:your )?(?:clothes|shirt|pants|dress|bra)\b", re.I), 0.85),
    (re.compile(r"\bshow me your (?:body|breasts|boobs|chest|ass|butt|private parts)\b", re.I), 0.90),
    (re.compile(r"\b(?:you are|you're|you look) (?:so )?(?:sexy|hot)\b", re.I), 0.60),
    (re.compile(r"\b(?:nice|beautiful|sexy|hot) (?:body|legs|breasts|boobs|ass|butt)\b", re.I), 0.80),
    (re.compile(r"\b(?:come here|come closer).{0,20}\b(?:kiss|touch|bed|sexy|hot)\b", re.I), 0.65),
)

SEXUAL_ADVANCE_CLINICAL_CONTEXT_RE = re.compile(
    r"\b(?:verbal sexual advances?|sexual advances?|sexual comments?|sexually inappropriate remarks?)\b",
    re.I,
)

STRANGE_NOISE_CONTEXT_RE = re.compile(
    r"\b(?:strange|weird|odd|unusual|non[-\s]?speech|nonverbal|non[-\s]?verbal|human|vocal|voice|audio)\s+"
    r"(?:noise|noises|sound|sounds|vocali[sz]ation|vocali[sz]ations|burst|bursts)\b",
    re.I,
)

STRANGE_NOISE_ANNOTATION_RE = re.compile(r"\[(?P<label>[^\]]+)\]|\((?P<paren>[^)]+)\)", re.I)

STRANGE_NOISE_DOCUMENTATION_RE = re.compile(
    r"\b(?:dataset|datasets|corpus|label|labels|class|classes|taxonomy|CMAI|includes?|contains?|classified as)\b",
    re.I,
)

COMPLAINT_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    ("tired_of_this", re.compile(r"\b(?:i am|i'm|im)\s+tired\s+of\s+(?:this|it|everything)\b", re.I), 0.70),
    ("dislike_this", re.compile(r"\bi\s+don'?t\s+like\s+(?:this|it|that|this place|this food)\b", re.I), 0.70),
    ("pain_or_discomfort", re.compile(r"\b(?:this\s+hurts|it\s+hurts|i'?m\s+uncomfortable|i\s+am\s+uncomfortable|this\s+is\s+uncomfortable)\b", re.I), 0.75),
    ("not_listened_to", re.compile(r"\bnobody\s+(?:listens|cares)(?:\s+to|\s+about)?\s+me\b", re.I), 0.80),
    ("why_bad_happening", re.compile(r"\bwhy\s+(?:is\s+this\s+happening|won'?t\s+anyone\s+help\s+me)\b", re.I), 0.75),
    ("terrible_or_wrong", re.compile(r"\b(?:this\s+is\s+terrible|everything\s+is\s+wrong|this\s+food\s+is\s+awful)\b", re.I), 0.75),
    ("dont_want_this", re.compile(r"\bi\s+don'?t\s+want\s+(?:this|it|that)\b", re.I), 0.70),
    ("too_noisy", re.compile(r"\bthis\s+place\s+is\s+too\s+noisy\b", re.I), 0.70),
    ("cant_stand", re.compile(r"\bi\s+can'?t\s+stand\s+(?:this|it|that|this\s+anymore|it\s+anymore)\b", re.I), 0.80),
)

COMPLAINT_KEYWORD_RE = re.compile(
    r"\b(?:tired of|don'?t like|hurts?|uncomfortable|nobody listens|nobody cares|terrible|wrong|awful|too noisy|can'?t stand|won'?t anyone help)\b",
    re.I,
)

NEGATIVISM_EMOTIONAL_STATE_RE = re.compile(
    r"\b(?:sad|depressed|lonely|grief|pain|fear|anxiety|frustrated|frustration|disappointed|disappointment|awful|terrible|hard|bad|miss)\b",
    re.I,
)

NEGATIVISM_TAXONOMY: tuple[tuple[str, float, tuple[re.Pattern[str], ...]], ...] = (
    (
        "refusal",
        0.75,
        (
            re.compile(r"\b(?:i|i['’]m|i am)\s+(?:won't|wont|will not|willn't)\b", re.I),
            re.compile(r"\b(?:i|i['’]m|i am)\s+(?:don'?t|do not)\s+(?:want(?:\s+to)?|like|need|care|go|take|do|listen|cooperate|touch|leave|come|talk)\b", re.I),
            re.compile(r"\b(?:i|i['’]m|i am)\s+refuse(?:s|d)?\b", re.I),
            re.compile(r"\b(?:i|i['’]m|i am)\s+not\b(?!\s+(?:sad|depressed|lonely|grief|pain|fear|anxiety|frustrated|frustration|disappointed|disappointment|awful|terrible|hard|bad|miss))", re.I),
            re.compile(r"(?<!\w)(?:don'?t|do not)(?!\w)", re.I),
            re.compile(r"(?<!\w)(?:no(?: way)?|nah)(?!\w)", re.I),
        ),
    ),
    (
        "resistance",
        0.65,
        (
            re.compile(r"\b(?:don'?t|do not)\s+(?:touch|bother|tell|leave|go|come|talk|disturb|make)\b", re.I),
            re.compile(r"\b(?:leave me alone|go away|get away|get lost|back off|please stop)\b", re.I),
            re.compile(r"\bstop\b", re.I),
        ),
    ),
    (
        "non_compliance",
        0.65,
        (
            re.compile(r"\b(?:i|i['’]m|i am)\s+not\s+(?:taking|going|staying|doing|leaving|eating|drinking|listening|cooperating)\b", re.I),
            re.compile(r"\b(?:i|i['’]m|i am)\s+not\s+going\b", re.I),
            re.compile(r"\b(?:i|i['’]m|i am)\s+staying\s+here\b", re.I),
            re.compile(r"\b(?:i|i['’]m|i am)\s+don'?t\s+want\s+to\b", re.I),
        ),
    ),
    (
        "defiance",
        0.75,
        (
            re.compile(r"\byou\s+(?:can't|cannot|can not)\s+make\s+me\b", re.I),
            re.compile(r"\b(?:don'?t|do not)\s+tell\s+me\s+what\s+to\s+do\b", re.I),
            re.compile(r"\bi\s+decide\b", re.I),
            re.compile(r"\bi\s+said\s+no\b", re.I),
        ),
    ),
)

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
    request_like_terms = REQUEST_TERMS | HELP_TERMS | STOP_TERMS | ESCAPE_TERMS
    return any(term in lower for term in request_like_terms)


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



def _has_complaint_semantics(text: str) -> bool:
    return any(pattern.search(text) for _, pattern, _ in COMPLAINT_PATTERNS)

def _sentence_fragments(text: str) -> list[str]:
    """Return normalized utterance fragments split on sentence punctuation."""
    return [
        normalized
        for fragment in re.split(r"[.!?;]+", text)
        if (normalized := _normalize(fragment))
    ]


# ---------------------------------------------------------------------------
# History record
# ---------------------------------------------------------------------------

class _TranscriptRecord:
    __slots__ = ("text", "timestamp", "is_question", "is_request", "is_complaint")

    def __init__(self, text: str, timestamp: float) -> None:
        self.text = text
        self.timestamp = timestamp
        self.is_question = _is_question(text)
        self.is_request = _is_request(text)
        self.is_complaint = _has_complaint_semantics(text)


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
        self._max_records = max_records
        self._histories: dict[int | str | None, Deque[_TranscriptRecord]] = {}
        self._history = self._history_for(None)

    def _history_for(self, speaker_id: int | str | None) -> Deque[_TranscriptRecord]:
        history = self._histories.get(speaker_id)
        if history is None:
            history = deque(maxlen=self._max_records)
            self._histories[speaker_id] = history
        return history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, utterance: Utterance) -> LinguisticFeatures:
        """Compute all linguistic features for a completed utterance.

        Updates the rolling history after analysis.
        """
        text = utterance.full_text
        history = self._history_for(utterance.speaker_id)
        logger.info("BEHAVIOUR_TRACE linguistic_input speaker=%s transcript=%r", utterance.speaker_id, text)
        if not text.strip():
            logger.info("BEHAVIOUR_TRACE linguistic_output empty_transcript=True")
            return LinguisticFeatures()

        self._prune_history(utterance.end_time, history)

        evidence: dict = {}

        confidences = [line.transcript_confidence for line in utterance.lines
                       if line.transcript_confidence is not None]
        transcript_confidence = float(np.clip(np.mean(confidences), 0.0, 1.0)) if confidences else None
        evidence["transcript"] = {
            "confidence": transcript_confidence,
            "quality": "reported" if transcript_confidence is not None else "not_reported",
        }

        # A. Repetition
        rep, q_rep, req_rep = self._repetition_scores(text, history)
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

        # G. Yelling cues
        yelling = self._yelling_score(text)
        evidence["yelling"] = yelling

        # H. Verbal sexual advances
        sexual_advance = self._sexual_advance_score(text)
        evidence["sexual_advance"] = sexual_advance

        # I. Complaining
        complaint_score, complaint_keywords, complaint_patterns, complaint_confidence = (
            self._complaint_details(text, neg_sentiment, history)
        )
        evidence["complaint"] = {
            "complaint_score": complaint_score,
            "complaint_keywords": complaint_keywords,
            "complaint_patterns_matched": complaint_patterns,
            "complaint_confidence": complaint_confidence,
        }

        # J. Negativism (refusal / resistance / non-compliance / defiance)
        negativism_score, negativism_categories, negativism_phrases = self._negativism_details(text)
        evidence["negativism"] = {
            "negativism_score": negativism_score,
            "categories": negativism_categories,
            "matched_phrases": negativism_phrases,
        }

        # K. Strange human noises
        strange_noise_score, strange_noise_labels, strange_noise_datasets = self._strange_noise_details(text)
        evidence["strange_noise"] = {
            "strange_noise_score": strange_noise_score,
            "matched_labels": strange_noise_labels,
            "source_datasets": strange_noise_datasets,
        }

        # Update history
        history.append(_TranscriptRecord(text, utterance.end_time))

        features = LinguisticFeatures(
            repetition_score=float(np.clip(rep, 0.0, 1.0)),
            question_repetition_score=float(np.clip(q_rep, 0.0, 1.0)),
            negative_sentiment=float(np.clip(neg_sentiment, 0.0, 1.0)),
            urgency_score=float(np.clip(urgency, 0.0, 1.0)),
            threat_score=float(np.clip(threat, 0.0, 1.0)),
            profanity_score=float(np.clip(profanity, 0.0, 1.0)),
            imperative_score=float(np.clip(imperative, 0.0, 1.0)),
            yelling_score=float(np.clip(yelling, 0.0, 1.0)),
            sexual_advance_score=float(np.clip(sexual_advance, 0.0, 1.0)),
            complaint_score=float(np.clip(complaint_score, 0.0, 1.0)),
            negativism_score=float(np.clip(negativism_score, 0.0, 1.0)),
            strange_noise_score=float(np.clip(strange_noise_score, 0.0, 1.0)),
            evidence=evidence,
        )
        logger.info(
            "BEHAVIOUR_TRACE linguistic_output repetition=%.3f question_repetition=%.3f negative=%.3f urgency=%.3f threat=%.3f profanity=%.3f imperative=%.3f yelling=%.3f sexual_advance=%.3f complaint_score=%.3f complaint_keywords=%s complaint_patterns_matched=%s complaint_confidence=%.3f",
            features.repetition_score,
            features.question_repetition_score,
            features.negative_sentiment,
            features.urgency_score,
            features.threat_score,
            features.profanity_score,
            features.imperative_score,
            features.yelling_score,
            features.sexual_advance_score,
            features.complaint_score,
            complaint_keywords,
            complaint_patterns,
            complaint_confidence,
        )
        logger.info(
            "BEHAVIOUR_TRACE linguistic_output negativism=%.3f categories=%s phrases=%s",
            features.negativism_score,
            negativism_categories,
            negativism_phrases,
        )
        logger.info(
            "BEHAVIOUR_TRACE linguistic_output strange_noise=%.3f labels=%s datasets=%s",
            features.strange_noise_score,
            strange_noise_labels,
            strange_noise_datasets,
        )
        return features

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _repetition_scores(
        self, text: str, history: Deque[_TranscriptRecord] | None = None
    ) -> tuple[float, float, float]:
        """Return (repetition_score, question_repetition_score, request_repetition_score)."""
        intra_rep, intra_q_rep, intra_req_rep = self._intra_utterance_repetition_scores(text)
        current_words = _content_words(text)
        current_3grams = _ngrams(current_words, 3)
        is_q = _is_question(text)
        is_req = _is_request(text)

        word_sims, phrase_sims, fuzzy_sims = [], [], []
        q_sims, req_sims = [], []

        for rec in (self._history if history is None else history):
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

        history_rep = (
            0.25 * _agg(word_sims)
            + 0.35 * _agg(phrase_sims)
            + 0.40 * _agg(fuzzy_sims)
        )
        rep = max(intra_rep, history_rep)
        q_rep = max(intra_q_rep, _agg(q_sims))
        req_rep = max(intra_req_rep, _agg(req_sims))

        return rep, q_rep, req_rep

    def _intra_utterance_repetition_scores(self, text: str) -> tuple[float, float, float]:
        fragments = _sentence_fragments(text)
        phrase_scores: list[float] = []
        q_scores: list[float] = []
        req_scores: list[float] = []

        for i, fragment in enumerate(fragments):
            count = 1
            for other in fragments[i + 1 :]:
                if _fuzzy_similarity(fragment, other) >= 0.92:
                    count += 1
            if count < 2:
                continue
            score = min(1.0, 0.55 + 0.15 * count)
            phrase_scores.append(score)
            if _is_question(fragment):
                q_scores.append(score)
            if _is_request(fragment):
                req_scores.append(score)

        token_score = self._repeated_token_sequence_score(text)
        return (
            max(phrase_scores + [token_score], default=0.0),
            max(q_scores + ([token_score] if token_score and _is_question(text) else []), default=0.0),
            max(req_scores + ([token_score] if token_score and _is_request(text) else []), default=0.0),
        )

    @staticmethod
    def _repeated_token_sequence_score(text: str) -> float:
        words = _normalize(text).split()
        if len(words) < 4:
            return 0.0

        best = 0.0
        max_n = min(6, len(words) // 2)
        for n in range(2, max_n + 1):
            counts: dict[tuple[str, ...], int] = {}
            for i in range(len(words) - n + 1):
                gram = tuple(words[i : i + n])
                if not any(word not in STOP_WORDS for word in gram):
                    continue
                counts[gram] = counts.get(gram, 0) + 1
            if counts:
                count = max(counts.values())
                if count >= 2:
                    best = max(best, min(1.0, 0.55 + 0.10 * count + 0.03 * n))
        return best

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
        if _PROFANITY_META_CONTEXT.search(text):
            return 0.0
        return min(1.0, sum(weight for pattern, weight in PROFANITY_PATTERNS if pattern.search(text)))

    def _yelling_score(self, text: str) -> float:
        normalized = _normalize(text)
        score = 0.0
        term_matches = sum(1 for term in YELLING_TERMS if term in normalized)
        if term_matches:
            score += min(0.8, 0.45 + 0.15 * term_matches)

        caps_words = ALL_CAPS_WORD_RE.findall(text)
        if caps_words and any(word.lower() in LOUD_INTERJECTIONS for word in caps_words):
            score += min(0.6, 0.25 + 0.10 * len(caps_words))

        exclamation_count = sum(len(match.group(0)) for match in EXCLAMATION_RE.finditer(text))
        if exclamation_count >= 2:
            score += min(0.45, 0.15 + 0.05 * exclamation_count)

        return min(1.0, score)

    def _sexual_advance_score(self, text: str) -> float:
        if SEXUAL_ADVANCE_CLINICAL_CONTEXT_RE.search(text):
            return 0.0
        return min(1.0, sum(weight for pattern, weight in SEXUAL_ADVANCE_PATTERNS if pattern.search(text)))


    def _complaint_details(
        self,
        text: str,
        negative_sentiment: float,
        history: Deque[_TranscriptRecord] | None = None,
    ) -> tuple[float, list[str], list[str], float]:
        """Score explicit complaint semantics without treating plain sadness as complaining."""
        matched: list[tuple[str, float]] = [
            (name, weight) for name, pattern, weight in COMPLAINT_PATTERNS if pattern.search(text)
        ]
        keywords = sorted({match.group(0).lower() for match in COMPLAINT_KEYWORD_RE.finditer(text)})
        if not matched:
            return 0.0, keywords, [], 0.0

        base = max(weight for _, weight in matched)
        repeated_recent = any(
            rec.is_complaint for rec in (self._history if history is None else history)
        )
        if repeated_recent:
            base += 0.15
        if negative_sentiment >= 0.20:
            base += 0.10
        score = min(1.0, base)
        return score, keywords, [name for name, _ in matched], score

    def _negativism_details(self, text: str) -> tuple[float, list[str], list[str]]:
        """Heuristically score oppositional refusal/resistance/non-compliance/defiance.

        The taxonomy is intentionally narrow: it targets explicit oppositional
        behaviour such as refusing care, resisting contact, refusing to comply,
        or openly defying instructions. It avoids generic negative sentiment.
        """
        lower = _normalize(text)
        if not lower:
            return 0.0, [], []

        categories: list[str] = []
        matched_phrases: list[str] = []
        aggregate_score = 0.0

        for category_name, base_weight, patterns in NEGATIVISM_TAXONOMY:
            category_match_count = 0
            category_phrases: list[str] = []
            for pattern in patterns:
                matches = list(pattern.finditer(text))
                if not matches:
                    continue
                count = len(matches)
                category_match_count += count
                category_phrases.extend(match.group(0).strip() for match in matches)
            if category_match_count == 0:
                continue

            category_score = min(1.0, base_weight + 0.15 * max(0, category_match_count - 1))
            aggregate_score += category_score
            categories.append(category_name)
            matched_phrases.extend(category_phrases)

        if not categories:
            return 0.0, [], []

        score = min(1.0, aggregate_score)
        return score, categories, sorted(set(matched_phrases))

    def _strange_noise_details(self, text: str) -> tuple[float, list[str], list[str]]:
        """Score non-speech human vocalization labels from public dataset taxonomies."""
        normalized = _normalize(text)
        if not normalized:
            return 0.0, [], []

        matched: list[tuple[str, float, tuple[str, ...]]] = []
        for label in STRANGE_NOISE_LABELS:
            if not label.map_to_strange_noise:
                continue
            if any(self._contains_label_variant(normalized, variant) for variant in label.variants):
                matched.append((label.canonical, label.confidence, label.datasets))

        context_match = STRANGE_NOISE_CONTEXT_RE.search(text) is not None
        annotation_match = any(
            bool(part and self._annotation_contains_strange_noise_label(part))
            for match in STRANGE_NOISE_ANNOTATION_RE.finditer(text)
            for part in (match.group("label"), match.group("paren"))
        )

        if not matched and context_match:
            matched.append(("non-speech human vocalization", 0.65, ("Dataset label taxonomy",)))

        if not matched:
            return 0.0, [], []

        if STRANGE_NOISE_DOCUMENTATION_RE.search(text) and not annotation_match:
            return 0.0, [], []

        labels = sorted({label for label, _, _ in matched})
        datasets = sorted({dataset for _, _, label_datasets in matched for dataset in label_datasets})
        base = max(confidence for _, confidence, _ in matched)
        if annotation_match:
            base += 0.10
        if len(labels) > 1:
            base += min(0.15, 0.05 * (len(labels) - 1))

        return min(1.0, base), labels, datasets

    @staticmethod
    def _contains_label_variant(normalized_text: str, variant: str) -> bool:
        normalized_variant = _normalize(variant)
        if not normalized_variant:
            return False
        return re.search(rf"(?<!\w){re.escape(normalized_variant)}(?!\w)", normalized_text) is not None

    def _annotation_contains_strange_noise_label(self, annotation: str) -> bool:
        normalized = _normalize(annotation)
        return any(
            label.map_to_strange_noise
            and any(self._contains_label_variant(normalized, variant) for variant in label.variants)
            for label in STRANGE_NOISE_LABELS
        )

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

    def _prune_history(
        self, current_time: float, history: Deque[_TranscriptRecord] | None = None
    ) -> None:
        target = self._history if history is None else history
        cutoff = current_time - self._history_sec
        while target and target[0].timestamp < cutoff:
            target.popleft()

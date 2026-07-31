"""Multi-label CMAI-inspired behaviour classifier.

Classifies observable audio behaviours from fused acoustic and
linguistic scores. Returns a list of canonical labels and supporting evidence
for each active label.

IMPORTANT: Only behaviours detectable from audio are classified.
Physical behaviours (pacing, restlessness, hitting, kicking, grabbing)
require video or wearable data and are NOT produced here.

The output is described as "CMAI-inspired" — not as a clinical CMAI
score. The system observes acoustic and linguistic cues only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import config
from audio_behaviour_taxonomy import build_behaviour_event, map_observed_behaviour
from event_models import AcousticFeatureWindow, FusedResult, LinguisticFeatures

logger = logging.getLogger(__name__)


@dataclass
class BehaviourLabel:
    """A single detected behaviour with supporting evidence."""

    label: str
    evidence: str
    confidence: float


def _canonical_label(label: str) -> str:
    if label == "No audio agitation detected":
        return label
    mapped = map_observed_behaviour(label)
    return mapped.canonical_label if mapped.mapping_status == "mapped" else "Unmapped audio behaviour"


# ---------------------------------------------------------------------------
# Individual rule implementations
# ---------------------------------------------------------------------------

def _check_screaming(
    result: FusedResult,
    acoustic: AcousticFeatureWindow | None,
) -> BehaviourLabel | None:
    """Screaming/shouting — acoustic-only, works even if transcript fails."""
    if acoustic is None:
        return None

    energy_contrib = result.acoustic_contributions.get("energy_above_baseline", 0.0)
    burst_contrib = result.acoustic_contributions.get("energy_burst", 0.0)

    energy_high = energy_contrib >= (config.ACOUSTIC_WEIGHTS["energy_z"] * config.BEHAVIOUR_ENERGY_Z_SHOUT / config.Z_CLIP)
    burst_high = burst_contrib >= (config.ACOUSTIC_WEIGHTS["energy_burst_z"] * config.BEHAVIOUR_ENERGY_BURST_SHOUT)
    voiced_present = acoustic.voiced_ratio >= 0.30

    if energy_high and burst_high and voiced_present:
        conf = min(1.0, (energy_contrib + burst_contrib) * 4.0)
        return BehaviourLabel(
            label=_canonical_label("Screaming/shouting"),
            evidence=(
                f"Energy far above baseline (contribution={energy_contrib:.3f}), "
                f"high energy burst (contribution={burst_contrib:.3f}), "
                f"voiced ratio={acoustic.voiced_ratio:.2f}"
            ),
            confidence=round(conf, 3),
        )
    return None


def _check_verbal_aggression(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    """Possible verbal aggression — requires high acoustics + threat/sentiment."""
    if linguistic is None:
        return None

    threat_ok = linguistic.threat_score >= config.BEHAVIOUR_VERBAL_AGGR_THREAT
    profanity_imperative_ok = (
        linguistic.profanity_score >= 0.30
        and linguistic.imperative_score >= 0.50
    )
    sentiment_ok = linguistic.negative_sentiment >= config.BEHAVIOUR_VERBAL_AGGR_SENTIMENT
    acoustic_ok = result.acoustic_score >= config.BEHAVIOUR_VERBAL_AGGR_ACOUSTIC

    if acoustic_ok and sentiment_ok and (threat_ok or profanity_imperative_ok):
        conf = min(1.0, (result.acoustic_score + linguistic.threat_score) / 2.0)
        triggers = []
        if threat_ok:
            triggers.append(f"threat score={linguistic.threat_score:.2f}")
        if profanity_imperative_ok:
            triggers.append("profanity+imperative")
        return BehaviourLabel(
            label=_canonical_label("Possible verbal aggression"),
            evidence=(
                f"Acoustic score={result.acoustic_score:.2f}, "
                f"negative sentiment={linguistic.negative_sentiment:.2f}, "
                + ", ".join(triggers)
            ),
            confidence=round(conf, 3),
        )
    return None


def _check_repetitive_verbalization(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    if linguistic.repetition_score >= config.BEHAVIOUR_REPETITION_THRESHOLD:
        return BehaviourLabel(
            label=_canonical_label("Repetitive verbalization"),
            evidence=f"Repetition score={linguistic.repetition_score:.2f} (threshold {config.BEHAVIOUR_REPETITION_THRESHOLD})",
            confidence=round(min(1.0, linguistic.repetition_score), 3),
        )
    return None


def _check_repeated_questioning(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    if linguistic.question_repetition_score >= config.BEHAVIOUR_Q_REP_THRESHOLD:
        return BehaviourLabel(
            label=_canonical_label("Repeated questioning"),
            evidence=f"Question repetition score={linguistic.question_repetition_score:.2f}",
            confidence=round(min(1.0, linguistic.question_repetition_score), 3),
        )
    return None


def _check_repeated_requests(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    """Repeated requests for attention (help, go home, give me, etc.)."""
    if linguistic is None:
        return None
    rep = linguistic.repetition_score
    urgency = linguistic.urgency_score
    req_rep = float(linguistic.evidence.get("repetition", {}).get("req_rep", 0.0))
    request_signal = max(urgency, req_rep)
    combined = 0.5 * rep + 0.5 * request_signal
    if combined >= config.BEHAVIOUR_REQUEST_REP_THRESHOLD:
        return BehaviourLabel(
            label=_canonical_label("Repeated requests"),
            evidence=(
                f"Combined repeated request score={combined:.2f}; "
                f"repetition={rep:.2f}, urgency={urgency:.2f}, request repetition={req_rep:.2f}"
            ),
            confidence=round(min(1.0, combined), 3),
        )
    return None


def _check_distressed_verbalization(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    if (
        linguistic.urgency_score >= config.BEHAVIOUR_URGENCY_THRESHOLD
        and (
            result.acoustic_score >= config.BEHAVIOUR_URGENCY_ACOUSTIC
            or result.reliability < 0.90
        )
    ):
        acoustic_component = result.acoustic_score if result.acoustic_features is not None else linguistic.urgency_score
        conf = min(1.0, (linguistic.urgency_score + acoustic_component) / 2.0)
        return BehaviourLabel(
            label=_canonical_label("Distressed/urgent verbalization"),
            evidence=(
                f"Urgency score={linguistic.urgency_score:.2f}, "
                f"acoustic score={result.acoustic_score:.2f}"
            ),
            confidence=round(conf, 3),
        )
    return None


class BehaviourClassifier:
    """Apply multi-label behaviour rules to a ``FusedResult``."""

    _RULES = [
        _check_screaming,
        _check_verbal_aggression,
        _check_repeated_requests,
        _check_repetitive_verbalization,
        _check_repeated_questioning,
        _check_distressed_verbalization,
    ]

    def classify(self, result: FusedResult) -> FusedResult:
        """Evaluate all rules and return an updated ``FusedResult``."""
        acoustic = result.acoustic_features
        linguistic = result.linguistic_features
        logger.info(
            "BEHAVIOUR_TRACE classifier_input transcript=%r severity=%s smoothed=%.3f acoustic_score=%.3f linguistic_score=%.3f acoustic_available=%s linguistic_available=%s",
            result.utterance.full_text if result.utterance else "",
            result.severity,
            result.smoothed_score,
            result.acoustic_score,
            result.linguistic_score,
            acoustic is not None,
            linguistic is not None,
        )
        if linguistic is not None:
            logger.info(
                "BEHAVIOUR_TRACE classifier_linguistic_features repetition=%.3f question_repetition=%.3f negative=%.3f urgency=%.3f threat=%.3f profanity=%.3f imperative=%.3f",
                linguistic.repetition_score,
                linguistic.question_repetition_score,
                linguistic.negative_sentiment,
                linguistic.urgency_score,
                linguistic.threat_score,
                linguistic.profanity_score,
                linguistic.imperative_score,
            )
        detected: list[BehaviourLabel] = []

        for rule in self._RULES:
            label = rule(result, acoustic if rule == _check_screaming else linguistic)
            if label is not None:
                existing = next((item for item in detected if item.label == label.label), None)
                if existing is None:
                    detected.append(label)
                elif label.confidence > existing.confidence:
                    detected[detected.index(existing)] = label
                logger.info("Behaviour detected: %s (confidence=%.3f)", label.label, label.confidence)

        if not detected and result.smoothed_score < config.SEVERITY_LOW_MAX:
            detected.append(
                BehaviourLabel(
                    label="No audio agitation detected",
                    evidence=f"Smoothed score={result.smoothed_score:.3f} below threshold {config.SEVERITY_LOW_MAX}",
                    confidence=1.0 - result.smoothed_score,
                )
            )

        result.behaviours = [b.label for b in detected]
        result.behaviour_events = []
        for behaviour in detected:
            if behaviour.label == "No audio agitation detected":
                continue
            event = build_behaviour_event(
                raw_behaviour=behaviour.label,
                timestamp=getattr(result.utterance, "end_time", None),
                notes=behaviour.evidence,
            )
            result.behaviour_events.append(event)

        for b in detected:
            result.linguistic_contributions[f"[{b.label}]"] = round(b.confidence, 4)

        logger.info(
            "BEHAVIOUR_TRACE classifier_output transcript=%r labels=%s event_labels=%s severity=%s contributions=%s",
            result.utterance.full_text if result.utterance else "",
            result.behaviours,
            [event.canonical_label for event in result.behaviour_events],
            result.severity,
            result.linguistic_contributions,
        )
        return result

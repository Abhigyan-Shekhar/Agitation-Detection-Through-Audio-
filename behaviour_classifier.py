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
import time
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
    """Screaming/shouting from acoustic energy, works even if transcript fails.

    Two detection paths:
    1. Z-score path — requires the fused acoustic score or relative-energy
       contributions to cross the configured thresholds.
    2. Absolute path — triggers directly on raw RMS/peak/clipping values that
       are physically impossible for normal conversational speech, bypassing
       z-score gating entirely.  This ensures sustained screaming is detected
       even when the rolling baseline has not converged.
    """
    if acoustic is None:
        return None

    energy_contrib = result.acoustic_contributions.get("energy_above_baseline", 0.0)
    burst_contrib = result.acoustic_contributions.get("energy_burst", 0.0)

    energy_high = energy_contrib >= (config.ACOUSTIC_WEIGHTS["energy_z"] * config.BEHAVIOUR_ENERGY_Z_SHOUT / config.Z_CLIP)
    burst_high = burst_contrib >= (config.ACOUSTIC_WEIGHTS["energy_burst_z"] * config.BEHAVIOUR_ENERGY_BURST_SHOUT)
    absolute_energy_high = (
        acoustic.rms_mean >= config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT
        and acoustic.rms_max >= config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT
    )
    clipping_high = (
        acoustic.clipping_ratio >= config.BEHAVIOUR_CLIPPING_SHOUT
        and acoustic.rms_mean >= config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT * 0.65
    )
    voiced_present = acoustic.voiced_ratio >= 0.30 or absolute_energy_high or clipping_high
    high_acoustic_agitation = result.acoustic_score >= config.BEHAVIOUR_VERBAL_AGGR_ACOUSTIC

    # ---- Debug gate trace -----------------------------------------------
    logger.debug(
        "SCREAM_GATE  rms_mean=%.4f rms_max=%.4f clipping=%.4f voiced=%.3f  "
        "energy_contrib=%.4f(need≥%.4f) burst_contrib=%.4f(need≥%.4f)  "
        "energy_high=%s burst_high=%s absolute_energy_high=%s clipping_high=%s "
        "voiced_present=%s high_acoustic_agitation=%s(need≥%.2f)",
        acoustic.rms_mean, acoustic.rms_max, acoustic.clipping_ratio, acoustic.voiced_ratio,
        energy_contrib, config.ACOUSTIC_WEIGHTS["energy_z"] * config.BEHAVIOUR_ENERGY_Z_SHOUT / config.Z_CLIP,
        burst_contrib, config.ACOUSTIC_WEIGHTS["energy_burst_z"] * config.BEHAVIOUR_ENERGY_BURST_SHOUT,
        energy_high, burst_high, absolute_energy_high, clipping_high,
        voiced_present, high_acoustic_agitation, config.BEHAVIOUR_VERBAL_AGGR_ACOUSTIC,
    )

    if voiced_present and ((energy_high and burst_high) or high_acoustic_agitation or absolute_energy_high or clipping_high):
        absolute_conf = max(
            min(1.0, acoustic.rms_mean / max(config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT, 1e-6) * 0.75),
            min(1.0, acoustic.rms_max / max(config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT, 1e-6) * 0.65),
            min(1.0, acoustic.clipping_ratio / max(config.BEHAVIOUR_CLIPPING_SHOUT, 1e-6) * 0.70),
        )
        conf = min(1.0, max((energy_contrib + burst_contrib) * 4.0, result.acoustic_score, absolute_conf))
        return BehaviourLabel(
            label=_canonical_label("Screaming/shouting"),
            evidence=(
                f"Energy far above baseline (contribution={energy_contrib:.3f}), "
                f"high energy burst (contribution={burst_contrib:.3f}), "
                f"voiced ratio={acoustic.voiced_ratio:.2f}, "
                f"rms_mean={acoustic.rms_mean:.3f}, rms_max={acoustic.rms_max:.3f}, "
                f"clipping={acoustic.clipping_ratio:.3f}"
            ),
            confidence=round(conf, 3),
        )
    return None


def _check_yelling_language(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    # Text can report or mention yelling ("stop yelling at me") but cannot
    # establish that the current microphone audio contains a scream.  Keep
    # this cue in diagnostics; screaming itself requires acoustic evidence.
    return None


def _check_verbal_aggression(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    """Cursing/verbal aggression from explicit profanity or threat + high acoustics."""
    if linguistic is None:
        return None

    if linguistic.profanity_score >= 0.50:
        transcript_quality = linguistic.evidence.get("transcript", {}).get("confidence")
        quality_factor = 1.0 if transcript_quality is None else max(0.55, float(transcript_quality))
        conf = min(1.0, max(linguistic.profanity_score * quality_factor, 0.55 + 0.15 * result.acoustic_score))
        return BehaviourLabel(
            label=_canonical_label("Possible verbal aggression"),
            evidence=(f"Explicit profanity score={linguistic.profanity_score:.2f}; "
                      f"ASR confidence={transcript_quality if transcript_quality is not None else 'not reported'}"),
            confidence=round(conf, 3),
        )

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


def _check_verbal_sexual_advances(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    if linguistic.sexual_advance_score >= 0.60:
        return BehaviourLabel(
            label=_canonical_label("Making verbal sexual advances"),
            evidence=f"Sexual advance score={linguistic.sexual_advance_score:.2f}",
            confidence=round(min(1.0, linguistic.sexual_advance_score), 3),
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


def _check_complaining(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    if linguistic.complaint_score >= config.BEHAVIOUR_COMPLAINT_THRESHOLD:
        details = linguistic.evidence.get("complaint", {}) if linguistic.evidence else {}
        patterns = details.get("complaint_patterns_matched", [])
        keywords = details.get("complaint_keywords", [])
        evidence = f"Complaint score={linguistic.complaint_score:.2f}"
        if patterns:
            evidence += f", patterns={patterns}"
        if keywords:
            evidence += f", keywords={keywords}"
        return BehaviourLabel(
            label=_canonical_label("Complaining"),
            evidence=evidence,
            confidence=round(min(1.0, linguistic.complaint_score), 3),
        )
    return None


def _check_negativism(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    if linguistic.negativism_score >= config.BEHAVIOUR_NEGATIVISM_THRESHOLD:
        details = linguistic.evidence.get("negativism", {}) if linguistic.evidence else {}
        categories = details.get("categories", [])
        phrases = details.get("matched_phrases", [])
        evidence = f"Negativism score={linguistic.negativism_score:.2f}"
        if categories:
            evidence += f", categories={categories}"
        if phrases:
            evidence += f", phrases={phrases}"
        return BehaviourLabel(
            label=_canonical_label("Negativism"),
            evidence=evidence,
            confidence=round(min(1.0, linguistic.negativism_score), 3),
        )
    return None


def _check_strange_noise(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    if linguistic.strange_noise_score >= config.BEHAVIOUR_STRANGE_NOISE_THRESHOLD:
        details = linguistic.evidence.get("strange_noise", {}) if linguistic.evidence else {}
        labels = details.get("matched_labels", [])
        datasets = details.get("source_datasets", [])
        evidence = f"Dataset-derived strange-noise score={linguistic.strange_noise_score:.2f}"
        if labels:
            evidence += f", labels={labels}"
        if datasets:
            evidence += f", sources={datasets}"
        return BehaviourLabel(
            label=_canonical_label("Making strange noises"),
            evidence=evidence,
            confidence=round(min(1.0, linguistic.strange_noise_score), 3),
        )
    return None


def _check_acoustic_strange_noise(
    result: FusedResult,
    acoustic: AcousticFeatureWindow | None,
) -> BehaviourLabel | None:
    if acoustic is None:
        return None
    if acoustic.non_speech_vocalization_score >= config.BEHAVIOUR_STRANGE_NOISE_THRESHOLD:
        label = acoustic.non_speech_vocalization_label or "non-speech human vocalization"
        evidence = (
            f"Raw-audio strange-noise score={acoustic.non_speech_vocalization_score:.2f}, "
            f"label={label}"
        )
        if acoustic.non_speech_vocalization_evidence:
            evidence += f", {acoustic.non_speech_vocalization_evidence}"
        return BehaviourLabel(
            label=_canonical_label("Making strange noises"),
            evidence=evidence,
            confidence=round(min(1.0, acoustic.non_speech_vocalization_score), 3),
        )
    return None


def _check_distressed_verbalization(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    if linguistic is None:
        return None
    transcript_confidence = linguistic.evidence.get("transcript", {}).get("confidence")
    transcript_reliable = (
        transcript_confidence is None
        or float(transcript_confidence) >= config.BEHAVIOUR_URGENCY_MIN_TRANSCRIPT_CONFIDENCE
    )
    # ASR confidence is evidence quality, not evidence of calmness.  Strong
    # urgent language may stand on its own when the transcript is reliable;
    # weaker text needs acoustic corroboration.
    if linguistic.urgency_score >= config.BEHAVIOUR_URGENCY_THRESHOLD and (
        transcript_reliable or result.acoustic_score >= config.BEHAVIOUR_URGENCY_ACOUSTIC
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
    """Apply multi-label behaviour rules to a ``FusedResult``.

    Screaming labels use hysteresis and temporal persistence: entering the
    screaming state requires multiple consecutive high-evidence windows, while
    leaving it uses a lower recovery threshold to avoid rapid oscillation.
    """

    _RULES = [
        _check_screaming,
        _check_yelling_language,
        _check_verbal_aggression,
        _check_verbal_sexual_advances,
        _check_repeated_requests,
        _check_repetitive_verbalization,
        _check_repeated_questioning,
        _check_complaining,
        _check_negativism,
        _check_strange_noise,
        _check_acoustic_strange_noise,
        _check_distressed_verbalization,
    ]

    def __init__(self) -> None:
        self._scream_positive_windows = 0
        self._scream_recovery_windows = 0
        self._scream_active = False
        self._scream_first_positive_ts: float | None = None
        self._last_scream_window_ts: float | None = None

    def _scream_gate(self, label: BehaviourLabel | None, result: FusedResult) -> BehaviourLabel | None:
        window_ts = getattr(result.acoustic_features, "end_time", None)
        # The dashboard refreshes every second while acoustic windows arrive
        # every 0.5 s. Never count the same acoustic window multiple times;
        # otherwise one loud word can satisfy a three-window persistence gate.
        if window_ts is not None and window_ts == self._last_scream_window_ts:
            return label if self._scream_active else None
        if window_ts is not None:
            self._last_scream_window_ts = window_ts
        if label is None:
            # Once confirmed, retain the active state through intermediate
            # evidence.  This is hysteresis, not a new detection: it stops a
            # sustained scream flickering off between feature windows.
            if self._scream_active and result.acoustic_score >= config.SCREAM_OFF_SCORE_THRESHOLD:
                return BehaviourLabel(
                    label=_canonical_label("Screaming/shouting"),
                    evidence=(f"Scream gate holding active state; acoustic score="
                              f"{result.acoustic_score:.2f} >= off threshold "
                              f"{config.SCREAM_OFF_SCORE_THRESHOLD:.2f}"),
                    confidence=round(result.acoustic_score, 3),
                )
            self._scream_positive_windows = 0
            self._scream_first_positive_ts = None
            if result.acoustic_score <= config.SCREAM_OFF_SCORE_THRESHOLD:
                self._scream_recovery_windows += 1
                if self._scream_recovery_windows >= config.SCREAM_RECOVERY_CONSECUTIVE_WINDOWS:
                    self._scream_active = False
            return None

        now = getattr(result.acoustic_features, "end_time", None) or time.time()
        gate_score = label.confidence
        if gate_score >= config.SCREAM_EXTREME_SCORE_THRESHOLD:
            self._scream_active = True
            self._scream_positive_windows = 1
            self._scream_recovery_windows = 0
            label.evidence += ", scream_gate=immediate_extreme_event"
            return label
        if gate_score >= config.SCREAM_ON_SCORE_THRESHOLD:
            self._scream_positive_windows += 1
            self._scream_recovery_windows = 0
            if self._scream_first_positive_ts is None:
                self._scream_first_positive_ts = now
        elif self._scream_active and gate_score >= config.SCREAM_OFF_SCORE_THRESHOLD:
            return label
        else:
            self._scream_positive_windows = 0
            self._scream_first_positive_ts = None
            self._scream_recovery_windows += 1
            return None

        duration = 0.0 if self._scream_first_positive_ts is None else max(0.0, now - self._scream_first_positive_ts)
        enough_windows = self._scream_positive_windows >= config.SCREAM_MIN_CONSECUTIVE_WINDOWS
        enough_duration = duration >= config.SCREAM_MIN_DURATION_SEC
        if self._scream_active or (enough_windows and enough_duration):
            self._scream_active = True
            label.evidence += (
                f", scream_gate=active positives={self._scream_positive_windows} "
                f"duration={duration:.2f}s on={config.SCREAM_ON_SCORE_THRESHOLD:.2f} "
                f"off={config.SCREAM_OFF_SCORE_THRESHOLD:.2f}"
            )
            return label

        logger.info(
            "Screaming candidate suppressed by persistence gate: positives=%d duration=%.2fs score=%.3f",
            self._scream_positive_windows, duration, result.acoustic_score,
        )
        return None

    @property
    def scream_debug_state(self) -> dict[str, object]:
        return {
            "active": self._scream_active,
            "consecutive_positive_windows": self._scream_positive_windows,
            "recovery_windows": self._scream_recovery_windows,
            "on_threshold": config.SCREAM_ON_SCORE_THRESHOLD,
            "off_threshold": config.SCREAM_OFF_SCORE_THRESHOLD,
            "min_consecutive_windows": config.SCREAM_MIN_CONSECUTIVE_WINDOWS,
            "last_distinct_window_end_time": self._last_scream_window_ts,
        }

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
                "BEHAVIOUR_TRACE classifier_linguistic_features repetition=%.3f question_repetition=%.3f negative=%.3f urgency=%.3f threat=%.3f profanity=%.3f imperative=%.3f yelling=%.3f sexual_advance=%.3f complaint_score=%.3f negativism=%.3f strange_noise=%.3f",
                linguistic.repetition_score,
                linguistic.question_repetition_score,
                linguistic.negative_sentiment,
                linguistic.urgency_score,
                linguistic.threat_score,
                linguistic.profanity_score,
                linguistic.imperative_score,
                linguistic.yelling_score,
                linguistic.sexual_advance_score,
                linguistic.complaint_score,
                linguistic.negativism_score,
                linguistic.strange_noise_score,
            )
        detected: list[BehaviourLabel] = []

        for rule in self._RULES:
            label = rule(result, acoustic if rule in (_check_screaming, _check_acoustic_strange_noise) else linguistic)
            if rule is _check_screaming:
                label = self._scream_gate(label, result)
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
                person=result.speaker_label,
                speaker_id=result.speaker_id,
                speaker_label=result.speaker_label,
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

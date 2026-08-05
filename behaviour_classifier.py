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


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ramp_score(value: float, floor: float, target: float) -> float:
    if target <= floor:
        return 1.0 if value >= target else 0.0
    return _clamp_unit((value - floor) / (target - floor))


def _scream_acoustic_scores(acoustic: AcousticFeatureWindow) -> dict[str, float]:
    """Return normalized acoustic cues that make screaming distinct from loud speech."""
    rms_score = _clamp_unit(acoustic.rms_mean / max(config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT, 1e-6))
    peak_score = _clamp_unit(acoustic.rms_max / max(config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT, 1e-6))
    clipping_score = _clamp_unit(acoustic.clipping_ratio / max(config.BEHAVIOUR_CLIPPING_SHOUT, 1e-6))
    energy_score = max(rms_score, 0.90 * peak_score, 0.85 * clipping_score)

    pitch_score = max(
        _ramp_score(acoustic.pitch_median, 180.0, config.BEHAVIOUR_SCREAM_PITCH_MEDIAN_HZ),
        _clamp_unit(acoustic.pitch_range / max(config.BEHAVIOUR_SCREAM_PITCH_RANGE_HZ, 1e-6)),
        _clamp_unit(acoustic.pitch_variance / max(config.BEHAVIOUR_SCREAM_PITCH_VARIANCE, 1e-6)),
    )

    centroid_score = _clamp_unit(acoustic.spectral_centroid / max(config.BEHAVIOUR_SCREAM_SPECTRAL_CENTROID_HZ, 1e-6))
    rolloff_score = _clamp_unit(acoustic.spectral_rolloff / max(config.BEHAVIOUR_SCREAM_SPECTRAL_ROLLOFF_HZ, 1e-6))
    zcr_score = _clamp_unit(acoustic.zcr_mean / max(config.BEHAVIOUR_SCREAM_ZCR, 1e-6))
    spectral_score = 0.45 * centroid_score + 0.35 * rolloff_score + 0.20 * zcr_score

    onset_score = _clamp_unit(max(0.0, acoustic.rms_slope) / max(config.BEHAVIOUR_SCREAM_ONSET_SLOPE, 1e-6))
    duration = max(0.0, acoustic.duration())
    duration_score = _clamp_unit(duration / max(config.BEHAVIOUR_SCREAM_SUSTAINED_DURATION_SEC, 1e-6))

    total = (
        0.35 * energy_score
        + 0.25 * pitch_score
        + 0.20 * spectral_score
        + 0.10 * onset_score
        + 0.10 * duration_score
    )
    return {
        "energy": energy_score,
        "pitch": pitch_score,
        "spectral": spectral_score,
        "onset": onset_score,
        "duration": duration_score,
        "total": _clamp_unit(total),
    }


# ---------------------------------------------------------------------------
# Individual rule implementations
# ---------------------------------------------------------------------------

def _check_screaming(
    result: FusedResult,
    acoustic: AcousticFeatureWindow | None,
) -> BehaviourLabel | None:
    """Screaming/shouting from combined acoustic evidence, even if transcript fails."""
    if acoustic is None:
        return None

    energy_contrib = result.acoustic_contributions.get("energy_above_baseline", 0.0)
    burst_contrib = result.acoustic_contributions.get("energy_burst", 0.0)
    cue_scores = _scream_acoustic_scores(acoustic)

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
    energy_gate = (
        acoustic.rms_mean >= config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT * config.BEHAVIOUR_SCREAM_MIN_RMS_RATIO
        or acoustic.rms_max >= config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT * config.BEHAVIOUR_SCREAM_MIN_PEAK_RATIO
        or clipping_high
    )
    burst_or_sustained_energy = (
        acoustic.rms_mean >= config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT * 0.75
        or acoustic.rms_max >= config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT * config.BEHAVIOUR_SCREAM_MIN_PEAK_RATIO
        or clipping_high
    )
    vocal_gate = (
        acoustic.voiced_ratio >= config.BEHAVIOUR_SCREAM_MIN_VOICED_RATIO
        or cue_scores["pitch"] >= 0.45
        or (acoustic.harmonic_to_noise_ratio > -5.0 and cue_scores["spectral"] < 0.95)
    )
    enough_duration = (
        acoustic.duration() >= config.BEHAVIOUR_SCREAM_MIN_DURATION_SEC
        or (cue_scores["onset"] >= 0.80 and acoustic.rms_max >= config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT)
    )
    scream_shape = (
        (cue_scores["pitch"] >= 0.55 and cue_scores["spectral"] >= 0.50)
        or (cue_scores["pitch"] >= 0.70 and cue_scores["onset"] >= 0.50)
        or (cue_scores["spectral"] >= 0.75 and cue_scores["onset"] >= 0.70 and acoustic.voiced_ratio >= 0.15)
        or (clipping_high and max(cue_scores["pitch"], cue_scores["spectral"]) >= 0.55)
        or (clipping_high and acoustic.rms_mean >= config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT * 1.50)
    )
    high_acoustic_agitation = result.acoustic_score >= config.BEHAVIOUR_VERBAL_AGGR_ACOUSTIC
    legacy_energy_only = (
        high_acoustic_agitation
        and absolute_energy_high
        and acoustic.voiced_ratio >= 0.60
        and acoustic.pitch_median == 0.0
        and acoustic.pitch_range == 0.0
        and acoustic.pitch_variance == 0.0
        and acoustic.spectral_centroid == 0.0
        and acoustic.spectral_rolloff == 0.0
        and acoustic.zcr_mean == 0.0
    )

    acoustic_specific_scream = (
        energy_gate
        and burst_or_sustained_energy
        and vocal_gate
        and enough_duration
        and scream_shape
        and cue_scores["total"] >= config.BEHAVIOUR_SCREAM_SCORE_THRESHOLD
    )
    clipped_saturation_scream = (
        clipping_high
        and acoustic.rms_mean >= config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT * 1.50
        and acoustic.rms_max >= config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT
    )

    if acoustic_specific_scream or clipped_saturation_scream or legacy_energy_only:
        absolute_conf = max(
            min(1.0, acoustic.rms_mean / max(config.BEHAVIOUR_ABSOLUTE_RMS_SHOUT, 1e-6) * 0.75),
            min(1.0, acoustic.rms_max / max(config.BEHAVIOUR_ABSOLUTE_PEAK_SHOUT, 1e-6) * 0.65),
            min(1.0, acoustic.clipping_ratio / max(config.BEHAVIOUR_CLIPPING_SHOUT, 1e-6) * 0.70),
        )
        conf = min(1.0, max((energy_contrib + burst_contrib) * 4.0, result.acoustic_score, absolute_conf, cue_scores["total"]))
        logger.info(
            "BEHAVIOUR_DEBUG screaming_rule triggered acoustic_rms=%.4f acoustic_peak=%.4f pitch=%.2f spectral=%.2f onset=%.2f duration=%.2f total=%.3f energy_gate=%s burst_or_sustained=%s vocal_gate=%s enough_duration=%s scream_shape=%s",
            acoustic.rms_mean,
            acoustic.rms_max,
            cue_scores["pitch"],
            cue_scores["spectral"],
            cue_scores["onset"],
            cue_scores["duration"],
            cue_scores["total"],
            energy_gate,
            burst_or_sustained_energy,
            vocal_gate,
            enough_duration,
            scream_shape,
        )
        return BehaviourLabel(
            label=_canonical_label("Screaming/shouting"),
            evidence=(
                f"Energy far above baseline (contribution={energy_contrib:.3f}), "
                f"high energy burst (contribution={burst_contrib:.3f}), "
                f"voiced ratio={acoustic.voiced_ratio:.2f}, "
                f"rms_mean={acoustic.rms_mean:.3f}, rms_max={acoustic.rms_max:.3f}, "
                f"clipping={acoustic.clipping_ratio:.3f}, "
                f"scream cues energy={cue_scores['energy']:.2f}, pitch={cue_scores['pitch']:.2f}, "
                f"spectral={cue_scores['spectral']:.2f}, onset={cue_scores['onset']:.2f}, "
                f"duration={cue_scores['duration']:.2f}, combined={cue_scores['total']:.2f}"
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
    if linguistic.yelling_score >= 0.50:
        return BehaviourLabel(
            label=_canonical_label("Screaming/shouting"),
            evidence=f"Transcript yelling score={linguistic.yelling_score:.2f}",
            confidence=round(min(1.0, linguistic.yelling_score), 3),
        )
    return None


def _check_verbal_aggression(
    result: FusedResult,
    linguistic: LinguisticFeatures | None,
) -> BehaviourLabel | None:
    """Cursing/verbal aggression from explicit profanity or threat + high acoustics."""
    if linguistic is None:
        return None

    if linguistic.profanity_score >= 0.50:
        conf = min(1.0, max(linguistic.profanity_score, 0.55 + 0.15 * result.acoustic_score))
        return BehaviourLabel(
            label=_canonical_label("Possible verbal aggression"),
            evidence=f"Explicit profanity score={linguistic.profanity_score:.2f}",
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
        _check_yelling_language,
        _check_verbal_aggression,
        _check_verbal_sexual_advances,
        _check_repeated_requests,
        _check_repetitive_verbalization,
        _check_repeated_questioning,
        _check_complaining,
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
                "BEHAVIOUR_TRACE classifier_linguistic_features repetition=%.3f question_repetition=%.3f negative=%.3f urgency=%.3f threat=%.3f profanity=%.3f imperative=%.3f yelling=%.3f sexual_advance=%.3f complaint_score=%.3f",
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

        if label is not None and label.label == "Screaming":
            logger.info(
                "BEHAVIOUR_DEBUG classifier_decision final_behaviours=%s detected_label=%s confidence=%.3f",
                [b.label for b in detected],
                label.label,
                label.confidence,
            )

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

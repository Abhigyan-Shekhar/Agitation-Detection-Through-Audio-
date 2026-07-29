"""Score fusion, temporal smoothing, and reliability estimation.

Responsibilities
----------------
* Accept aggregated ``AcousticFeatureWindow`` and ``LinguisticFeatures``
  for a completed utterance.
* Z-score each acoustic feature via ``BaselineManager``.
* Compute the acoustic branch score via a sigmoid weighted sum.
* Compute the linguistic branch score as a linear weighted sum.
* Fuse into a raw final score (60% acoustic, 40% linguistic).
* Apply asymmetric EMA smoothing (fast escalation, slow de-escalation).
* Compute a reliability estimate.
* Return a ``FusedResult`` with per-feature contribution breakdown for
  the explainability panel.

Design
------
* The ``ScoreFusion`` instance is long-lived (held in Streamlit session
  state) so it retains EMA state between utterances.
* Thread-safety: called from the Streamlit fragment thread only.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

import config
from baseline_manager import BaselineManager
from event_models import (
    AcousticFeatureWindow,
    FusedResult,
    LinguisticFeatures,
    Utterance,
)

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class ScoreFusion:
    """Stateful fusion engine — retains EMA state across utterances.

    Parameters
    ----------
    baseline_manager:
        Shared ``BaselineManager`` instance used for z-score normalisation.
    """

    def __init__(self, baseline_manager: BaselineManager) -> None:
        self._bm = baseline_manager
        self._prev_smoothed: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        utterance: Utterance,
        acoustic: AcousticFeatureWindow | None,
        linguistic: LinguisticFeatures,
    ) -> FusedResult:
        """Compute and return a ``FusedResult`` for one utterance."""

        # ---- Acoustic branch ----------------------------------------
        acoustic_score, acoustic_contributions = self._acoustic_score(acoustic)

        # ---- Linguistic branch --------------------------------------
        linguistic_score, linguistic_contributions = self._linguistic_score(linguistic)

        # ---- Raw fusion ----------------------------------------------
        raw_final = (
            config.ACOUSTIC_FUSION_WEIGHT * acoustic_score
            + config.LINGUISTIC_FUSION_WEIGHT * linguistic_score
        )
        raw_final = _clamp(raw_final)

        # ---- Asymmetric EMA smoothing --------------------------------
        if raw_final > self._prev_smoothed:
            alpha = config.EMA_ALPHA_UP
        else:
            alpha = config.EMA_ALPHA_DOWN
        smoothed = alpha * raw_final + (1 - alpha) * self._prev_smoothed
        smoothed = _clamp(smoothed)
        self._prev_smoothed = smoothed

        # ---- Reliability --------------------------------------------
        reliability = self._reliability(acoustic, linguistic, acoustic_score, linguistic_score)

        # ---- Severity -----------------------------------------------
        severity = self._severity(smoothed)

        trace = utterance.latency_trace
        if trace is not None:
            trace.inference_ts = time.monotonic()

        logger.info(
            "Fused — acoustic=%.3f linguistic=%.3f raw=%.3f smoothed=%.3f severity=%s reliability=%.2f latency=%s",
            acoustic_score, linguistic_score, raw_final, smoothed, severity, reliability,
            trace.durations_ms() if trace else {},
        )
        logger.info(
            "BEHAVIOUR_TRACE fusion_output transcript=%r acoustic=%.3f linguistic=%.3f raw=%.3f smoothed=%.3f severity=%s reliability=%.3f acoustic_available=%s",
            utterance.full_text,
            acoustic_score,
            linguistic_score,
            raw_final,
            smoothed,
            severity,
            reliability,
            acoustic is not None,
        )

        return FusedResult(
            acoustic_score=round(acoustic_score, 4),
            linguistic_score=round(linguistic_score, 4),
            raw_final_score=round(raw_final, 4),
            smoothed_score=round(smoothed, 4),
            severity=severity,
            reliability=round(reliability, 4),
            behaviours=[],   # filled in by BehaviourClassifier
            acoustic_contributions=acoustic_contributions,
            linguistic_contributions=linguistic_contributions,
            utterance=utterance,
            acoustic_features=acoustic,
            linguistic_features=linguistic,
            latency_trace=trace,
        )

    def reset(self) -> None:
        """Reset EMA state (e.g. when microphone restarts)."""
        self._prev_smoothed = 0.0

    # ------------------------------------------------------------------
    # Acoustic branch
    # ------------------------------------------------------------------

    def _acoustic_score(
        self, acoustic: AcousticFeatureWindow | None
    ) -> tuple[float, dict[str, float]]:
        if acoustic is None:
            return 0.0, {}

        bm = self._bm

        # Z-score each feature (clamped to ±Z_CLIP by BaselineManager)
        energy_z = bm.z_score("rms_mean", acoustic.rms_mean)
        energy_max_z = bm.z_score("rms_max", acoustic.rms_max)
        pitch_range_z = bm.z_score("pitch_range", acoustic.pitch_range)
        pitch_var_z = bm.z_score("pitch_variance", acoustic.pitch_variance)

        # Speech rate approximation: voiced_ratio as proxy until word
        # timestamps are integrated. Replace with WPM when available.
        speech_rate_z = bm.z_score("voiced_ratio", acoustic.voiced_ratio)

        # Pause irregularity: high pause_ratio relative to baseline can be
        # agitation (broken speech, gasping) or calm silence — use cautiously
        pause_irr_z = bm.z_score("pause_ratio", acoustic.pause_ratio)

        weights = config.ACOUSTIC_WEIGHTS
        weighted_sum = (
            weights["energy_z"] * energy_z
            + weights["energy_burst_z"] * energy_max_z
            + weights["pitch_range_z"] * pitch_range_z
            + weights["pitch_variance_z"] * pitch_var_z
            + weights["speech_rate_z"] * speech_rate_z
            + weights["pause_irregularity_z"] * pause_irr_z
        )

        score = _clamp(_sigmoid(weighted_sum))

        contributions = {
            "energy_above_baseline": round(weights["energy_z"] * energy_z, 4),
            "energy_burst": round(weights["energy_burst_z"] * energy_max_z, 4),
            "pitch_range": round(weights["pitch_range_z"] * pitch_range_z, 4),
            "pitch_variation": round(weights["pitch_variance_z"] * pitch_var_z, 4),
            "speech_rate": round(weights["speech_rate_z"] * speech_rate_z, 4),
            "pause_irregularity": round(weights["pause_irregularity_z"] * pause_irr_z, 4),
        }

        return score, contributions

    # ------------------------------------------------------------------
    # Linguistic branch
    # ------------------------------------------------------------------

    def _linguistic_score(
        self, linguistic: LinguisticFeatures
    ) -> tuple[float, dict[str, float]]:
        w = config.LINGUISTIC_WEIGHTS
        raw = (
            w["repetition_score"] * linguistic.repetition_score
            + w["question_repetition_score"] * linguistic.question_repetition_score
            + w["negative_sentiment"] * linguistic.negative_sentiment
            + w["urgency_score"] * linguistic.urgency_score
            + w["threat_score"] * linguistic.threat_score
            + w["profanity_score"] * linguistic.profanity_score
        )
        score = _clamp(raw)

        contributions = {
            "repeated_phrases": round(w["repetition_score"] * linguistic.repetition_score, 4),
            "repeated_questions": round(w["question_repetition_score"] * linguistic.question_repetition_score, 4),
            "negative_sentiment": round(w["negative_sentiment"] * linguistic.negative_sentiment, 4),
            "urgency_language": round(w["urgency_score"] * linguistic.urgency_score, 4),
            "threat_language": round(w["threat_score"] * linguistic.threat_score, 4),
            "profanity": round(w["profanity_score"] * linguistic.profanity_score, 4),
        }
        return score, contributions

    # ------------------------------------------------------------------
    # Reliability
    # ------------------------------------------------------------------

    def _reliability(
        self,
        acoustic: AcousticFeatureWindow | None,
        linguistic: LinguisticFeatures,
        acoustic_score: float,
        linguistic_score: float,
    ) -> float:
        penalty = 0.0

        # Missing personal baseline
        penalty += self._bm.missing_baseline_penalty()

        # Clipping (microphone saturation)
        if acoustic:
            clipping = acoustic.clipping_ratio
            if clipping > 0.10:
                penalty += config.RELIABILITY_CLIPPING_PENALTY * min(1.0, clipping / 0.5)

        # Very short utterance (less content to analyse)
        if acoustic and acoustic.voiced_ratio < 0.20:
            penalty += config.RELIABILITY_SHORT_UTTERANCE_PENALTY

        # Branch disagreement
        disagreement = abs(acoustic_score - linguistic_score)
        if disagreement > 0.35:
            penalty += config.RELIABILITY_BRANCH_DISAGREEMENT_PENALTY * (disagreement / 1.0)

        return _clamp(1.0 - penalty)

    # ------------------------------------------------------------------
    # Severity
    # ------------------------------------------------------------------

    @staticmethod
    def _severity(score: float) -> str:
        if score < config.SEVERITY_LOW_MAX:
            return "Low"
        if score < config.SEVERITY_MILD_MAX:
            return "Mild"
        if score < config.SEVERITY_MODERATE_MAX:
            return "Moderate"
        return "High"

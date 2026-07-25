"""Canonical audio-only behaviour taxonomy for the agitation prototype.

This module is the single source of truth for audio-observable behaviour
normalisation and CMAI mapping. Gemini may suggest behaviours, but this layer
applies deterministic, review-aware mapping before the behaviour is surfaced in
structured events or the dashboard. The current CMAI category strings remain
subject to stakeholder verification against the source CMAI sheet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from event_models import BehaviourEvent


@dataclass(frozen=True)
class BehaviourTaxonomyEntry:
    """One canonical audio behaviour supported by the current prototype."""

    internal_code: str
    canonical_label: str
    cmai_category: str
    modality: str
    description: str
    aliases: tuple[str, ...] = ()
    evidence_source_type: str = "model_observation"


@dataclass(frozen=True)
class MappedBehaviour:
    """Deterministic mapping outcome for a raw behaviour string."""

    raw_detected_behaviour: str
    internal_code: str | None
    canonical_label: str
    cmai_category: str
    modality: str
    mapping_status: str
    description: str


SUPPORTED_AUDIO_BEHAVIOURS: tuple[BehaviourTaxonomyEntry, ...] = (
    BehaviourTaxonomyEntry(
        internal_code="AUDIO_SCREAMING",
        canonical_label="Screaming",
        cmai_category="Verbally agitated: screaming/shouting",
        modality="audio",
        description="Loud shouting or screaming that is clearly above normal speech.",
        aliases=(
            "scream",
            "screaming",
            "shout",
            "shouting",
            "yell",
            "yelling",
            "shriek",
        ),
    ),
    BehaviourTaxonomyEntry(
        internal_code="AUDIO_CURSING",
        canonical_label="Cursing / verbal aggression",
        cmai_category="Verbally agitated: verbal aggression",
        modality="audio",
        description="Profanity, insults, or overtly aggressive verbal language.",
        aliases=(
            "curse",
            "cursing",
            "swear",
            "swearing",
            "verbal aggression",
            "verbally aggressive",
            "profanity",
            "profane language",
            "hostile language",
            "insult",
            "insulting",
        ),
    ),
    BehaviourTaxonomyEntry(
        internal_code="AUDIO_REPETITIVE",
        canonical_label="Repetitive sentences or questions",
        cmai_category="Verbally non-aggressive: repetitive questioning",
        modality="audio",
        description="Repeated or looped verbal content, including repeated questions.",
        aliases=(
            "repetitive sentences or questions",
            "repetitive questioning",
            "repeated questioning",
            "repeated questions",
            "repetitive verbalisation",
            "repetitive verbalization",
            "repeated verbal behaviour",
            "repeated verbalization",
            "repeated verbalisation",
            "repetitive verbal behaviour",
            "repetitive sentence",
            "repeated sentence",
        ),
    ),
    BehaviourTaxonomyEntry(
        internal_code="AUDIO_STRANGE_NOISE",
        canonical_label="Making strange noises",
        cmai_category="Verbally non-aggressive: strange noises",
        modality="audio",
        description="Unusual or odd vocal sounds that are not normal speech.",
        aliases=(
            "strange noises",
            "strange noise",
            "weird noise",
            "weird noises",
            "odd noises",
            "odd noise",
            "unusual vocalization",
            "unusual vocalisation",
            "non speech vocalization",
            "non speech vocalisation",
            "grunting",
            "groaning",
            "moaning",
        ),
    ),
    BehaviourTaxonomyEntry(
        internal_code="AUDIO_COMPLAINING",
        canonical_label="Complaining",
        cmai_category="Verbally non-aggressive: complaining",
        modality="audio",
        description="Complaints or expressions of dissatisfaction.",
        aliases=(
            "complaining",
            "complain",
            "grumbling",
            "grumble",
            "complaint",
            "complaints",
        ),
    ),
    BehaviourTaxonomyEntry(
        internal_code="AUDIO_NEGATIVISM",
        canonical_label="Negativism",
        cmai_category="Verbally non-aggressive: negativism/refusal",
        modality="audio",
        description="Negative, rejecting, or oppositional verbal responses.",
        aliases=(
            "negativism",
            "refusal",
            "refuse",
            "refusing",
            "oppositional",
            "resistance",
        ),
    ),
    BehaviourTaxonomyEntry(
        internal_code="AUDIO_CONSTANT_REQUEST",
        canonical_label="Constant unwarranted requests for attention/help",
        cmai_category="Verbally non-aggressive: repeated requests for attention/help",
        modality="audio",
        description="Repeated requests for attention, help, or assistance.",
        aliases=(
            "constant requests",
            "constant request",
            "repeated requests",
            "repeated request",
            "requests for attention",
            "attention requests",
        ),
    ),
)


def get_supported_behaviours() -> tuple[BehaviourTaxonomyEntry, ...]:
    """Return the supported canonical audio behaviours."""
    return SUPPORTED_AUDIO_BEHAVIOURS


def _normalise_text(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    text = text.replace("’", "'").replace("–", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _matches_entry(text: str, entry: BehaviourTaxonomyEntry) -> bool:
    normalized = _normalise_text(text)
    if not normalized:
        return False

    alias_strings = { _normalise_text(alias) for alias in entry.aliases }
    if normalized in alias_strings:
        return True

    if entry.internal_code == "AUDIO_SCREAMING":
        return _contains_any(normalized, ("scream", "shout", "yell", "shriek"))

    if entry.internal_code == "AUDIO_CURSING":
        return _contains_any(
            normalized,
            (
                "curse",
                "swear",
                "verbal aggression",
                "verbally aggressive",
                "profanity",
                "profane language",
                "hostile language",
                "insult",
            ),
        )

    if entry.internal_code == "AUDIO_REPETITIVE":
        has_repeat = _contains_any(normalized, ("repetitive", "repeated", "repetition"))
        has_question_or_verbal = _contains_any(
            normalized,
            ("question", "questions", "questioning", "verbal", "sentence", "sentences", "behaviour", "behavior"),
        )
        return has_repeat and has_question_or_verbal

    if entry.internal_code == "AUDIO_STRANGE_NOISE":
        has_noise = _contains_any(normalized, ("noise", "noises"))
        has_strange = _contains_any(normalized, ("strange", "weird"))
        has_vocal = _contains_any(normalized, ("groan", "grunt", "moan"))
        return (has_noise and has_strange) or has_vocal

    if entry.internal_code == "AUDIO_COMPLAINING":
        return _contains_any(normalized, ("complain", "grumble", "complaint"))

    if entry.internal_code == "AUDIO_NEGATIVISM":
        return _contains_any(normalized, ("negativism", "refusal", "refuse", "refusing", "oppositional", "resistance"))

    if entry.internal_code == "AUDIO_CONSTANT_REQUEST":
        has_request = _contains_any(normalized, ("request", "requests", "help", "attention"))
        has_repeat_or_constant = _contains_any(normalized, ("constant", "repeated", "repeatedly", "repetition"))
        return has_request and has_repeat_or_constant

    return False


def map_observed_behaviour(raw_behaviour: str | None) -> MappedBehaviour:
    """Map a raw behaviour observation to the canonical audio taxonomy."""
    raw_text = raw_behaviour if isinstance(raw_behaviour, str) else str(raw_behaviour or "")
    if not raw_text.strip():
        return MappedBehaviour(
            raw_detected_behaviour=raw_text,
            internal_code=None,
            canonical_label="Unmapped audio behaviour",
            cmai_category="Unmapped audio behaviour",
            modality="audio",
            mapping_status="review_required",
            description="No behaviour text was supplied.",
        )

    if any(term in _normalise_text(raw_text) for term in ("pitch", "variance", "acoustic", "rms", "spectral", "voiced", "zcr")):
        return MappedBehaviour(
            raw_detected_behaviour=raw_text,
            internal_code=None,
            canonical_label="Unmapped audio behaviour",
            cmai_category="Unmapped audio behaviour",
            modality="audio",
            mapping_status="review_required",
            description="Acoustic-feature descriptions are not CMAI behaviours.",
        )

    physical_terms = ("pacing", "restlessness", "hitting", "kicking", "pushing", "grabbing", "walking", "leaving")
    if any(term in _normalise_text(raw_text) for term in physical_terms):
        return MappedBehaviour(
            raw_detected_behaviour=raw_text,
            internal_code=None,
            canonical_label="Unmapped audio behaviour",
            cmai_category="Unmapped audio behaviour",
            modality="audio",
            mapping_status="review_required",
            description="Physical behaviours require non-audio evidence and are not mapped by the audio-only layer.",
        )

    for entry in SUPPORTED_AUDIO_BEHAVIOURS:
        if _matches_entry(raw_text, entry):
            return MappedBehaviour(
                raw_detected_behaviour=raw_text,
                internal_code=entry.internal_code,
                canonical_label=entry.canonical_label,
                cmai_category=entry.cmai_category,
                modality=entry.modality,
                mapping_status="mapped",
                description=entry.description,
            )

    normalised = _normalise_text(raw_text)
    if not normalised or any(term in normalised for term in ("speak", "speech", "talk", "conversation", "normal", "weather", "hello")):
        return MappedBehaviour(
            raw_detected_behaviour=raw_text,
            internal_code=None,
            canonical_label="Unmapped audio behaviour",
            cmai_category="Unmapped audio behaviour",
            modality="audio",
            mapping_status="review_required",
            description="The text did not match a supported audio behaviour and was left for review.",
        )

    return MappedBehaviour(
        raw_detected_behaviour=raw_text,
        internal_code=None,
        canonical_label="Unmapped audio behaviour",
        cmai_category="Unmapped audio behaviour",
        modality="audio",
        mapping_status="review_required",
        description="The text did not match a supported audio behaviour and was left for review.",
    )


def map_behaviours_to_cmai(behaviours: list[str] | None) -> list[dict[str, Any]]:
    """Deterministically map raw behaviour observations to canonical taxonomy."""
    mappings: list[dict[str, Any]] = []
    for behaviour in behaviours or []:
        mapped = map_observed_behaviour(behaviour)
        mappings.append(
            {
                "behaviour": behaviour,
                "internal_code": mapped.internal_code,
                "canonical_label": mapped.canonical_label,
                "cmai_category": mapped.cmai_category,
                "modality": mapped.modality,
                "mapping_status": mapped.mapping_status,
                "raw_detected_behaviour": mapped.raw_detected_behaviour,
            }
        )
    return mappings


def build_behaviour_event(
    raw_behaviour: str | None,
    *,
    person: str | None = None,
    timestamp: Any = None,
    location: str | None = None,
    severity: str | None = None,
    duration: float | None = None,
    trigger: str | None = None,
    intervention: str | None = None,
    outcome: str | None = None,
    notes: str | None = None,
    modality: str = "audio",
) -> BehaviourEvent:
    """Create a structured behaviour event from a raw observation."""
    mapping = map_observed_behaviour(raw_behaviour)
    return BehaviourEvent(
        event_id=f"behaviour-{uuid4().hex[:8]}",
        internal_code=mapping.internal_code,
        behaviour_type=mapping.canonical_label if mapping.mapping_status == "mapped" else "Unmapped audio behaviour",
        canonical_label=mapping.canonical_label,
        cmai_category=mapping.cmai_category,
        person=person,
        timestamp=timestamp,
        location=location,
        severity=severity,
        duration=duration,
        trigger=trigger,
        intervention=intervention,
        outcome=outcome,
        notes=notes,
        modality=modality,
        raw_detected_behaviour=mapping.raw_detected_behaviour,
        mapping_status=mapping.mapping_status,
    )

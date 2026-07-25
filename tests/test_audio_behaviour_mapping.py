from __future__ import annotations

from person3_module import analyze_person3, map_behaviours_to_cmai
from audio_behaviour_taxonomy import (
    BehaviourEvent,
    build_behaviour_event,
    map_observed_behaviour,
    get_supported_behaviours,
)


def test_supported_audio_behaviours_have_expected_canonical_labels():
    taxonomy = get_supported_behaviours()
    assert {entry.internal_code for entry in taxonomy} == {
        "AUDIO_SCREAMING",
        "AUDIO_CURSING",
        "AUDIO_REPETITIVE",
        "AUDIO_STRANGE_NOISE",
        "AUDIO_COMPLAINING",
        "AUDIO_NEGATIVISM",
        "AUDIO_CONSTANT_REQUEST",
    }


def test_aliases_normalise_to_the_same_canonical_behaviour():
    aliases = [
        "repetitive questioning",
        "Repeated questions",
        "repetitive verbalisation",
        "repeated verbal behaviour",
    ]
    for alias in aliases:
        mapped = map_observed_behaviour(alias)
        assert mapped.internal_code == "AUDIO_REPETITIVE"
        assert mapped.canonical_label == "Repetitive sentences or questions"


def test_capitalisation_and_whitespace_are_ignored():
    mapped = map_observed_behaviour("  SCREAMING!!!  ")
    assert mapped.internal_code == "AUDIO_SCREAMING"
    assert mapped.canonical_label == "Screaming"


def test_normal_speech_is_left_unmapped():
    mapped = map_observed_behaviour("The patient is speaking normally about the weather")
    assert mapped.mapping_status == "review_required"
    assert mapped.canonical_label == "Unmapped audio behaviour"


def test_acoustic_features_do_not_map_to_cmai_behaviours():
    mapped = map_observed_behaviour("elevated pitch variance and high rms energy")
    assert mapped.mapping_status == "review_required"
    assert mapped.internal_code is None


def test_negative_emotional_language_does_not_map_to_negativism():
    mapped = map_observed_behaviour("negative emotional language")
    assert mapped.mapping_status == "review_required"
    assert mapped.internal_code != "AUDIO_NEGATIVISM"


def test_refusal_maps_to_negativism():
    mapped = map_observed_behaviour("refusal")
    assert mapped.internal_code == "AUDIO_NEGATIVISM"


def test_physical_aggression_does_not_map_to_cursing():
    mapped = map_observed_behaviour("physical aggression")
    assert mapped.mapping_status == "review_required"
    assert mapped.internal_code != "AUDIO_CURSING"


def test_verbal_aggression_maps_to_cursing():
    mapped = map_observed_behaviour("verbal aggression")
    assert mapped.internal_code == "AUDIO_CURSING"


def test_profanity_maps_to_cursing():
    mapped = map_observed_behaviour("profanity")
    assert mapped.internal_code == "AUDIO_CURSING"


def test_unusual_vocalization_maps_to_strange_noises():
    mapped = map_observed_behaviour("unusual vocalization")
    assert mapped.internal_code == "AUDIO_STRANGE_NOISE"


def test_help_me_alone_does_not_map_to_constant_requests():
    mapped = map_observed_behaviour("help me")
    assert mapped.mapping_status == "review_required"
    assert mapped.internal_code != "AUDIO_CONSTANT_REQUEST"


def test_repeated_requests_for_help_map_to_constant_requests():
    mapped = map_observed_behaviour("repeated requests for help")
    assert mapped.internal_code == "AUDIO_CONSTANT_REQUEST"


def test_physical_behaviours_remain_review_required():
    mapped = map_observed_behaviour("pacing back and forth")
    assert mapped.mapping_status == "review_required"
    assert mapped.raw_detected_behaviour == "pacing back and forth"


def test_acoustic_feature_descriptions_remain_review_required():
    mapped = map_observed_behaviour("elevated pitch variance and high rms energy")
    assert mapped.mapping_status == "review_required"
    assert mapped.internal_code is None


def test_unknown_gemini_behaviour_preserves_raw_text_and_marks_review_required():
    mapped = map_observed_behaviour("mysterious humming from the hallway")
    assert mapped.mapping_status == "review_required"
    assert mapped.raw_detected_behaviour == "mysterious humming from the hallway"


def test_multiple_behaviours_are_mapped_into_structured_events():
    raw_behaviours = ["repetitive questioning", "screaming"]
    mappings = map_behaviours_to_cmai(raw_behaviours)
    assert len(mappings) == 2
    assert mappings[0]["internal_code"] == "AUDIO_REPETITIVE"
    assert mappings[1]["internal_code"] == "AUDIO_SCREAMING"


def test_behaviour_event_creation_uses_unknown_optional_metadata():
    event = build_behaviour_event(
        raw_behaviour="complaining loudly",
        person=None,
        timestamp=None,
        location=None,
        severity=None,
        duration=None,
        trigger=None,
        intervention=None,
        outcome=None,
        notes=None,
    )
    assert isinstance(event, BehaviourEvent)
    assert event.internal_code == "AUDIO_COMPLAINING"
    assert event.canonical_label == "Complaining"
    assert event.mapping_status == "mapped"
    assert event.person is None
    assert event.location is None
    assert event.trigger is None
    assert event.intervention is None
    assert event.outcome is None


def test_person3_mapping_keeps_review_required_behaviours():
    class FakeAnalyzer:
        def analyze(self, transcript, acoustic_features, acoustic_score=None):
            return {
                "emotion": "neutral",
                "agitation_score": 0.1,
                "behaviours": ["The resident is speaking normally and then says something odd"],
                "reasoning": "No clear observable behaviour detected.",
            }

    result = analyze_person3(
        "The resident is speaking normally and then says something odd",
        {"rms_energy": 0.1},
        analyzer=FakeAnalyzer(),
    )
    assert result["cmai_mapping"]
    assert result["cmai_mapping"][0]["mapping_status"] == "review_required"
    assert result["cmai_mapping"][0]["raw_detected_behaviour"] == "The resident is speaking normally and then says something odd"

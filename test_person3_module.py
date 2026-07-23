import json

import pytest

from person3_module import GeminiBehaviourAnalyzer, analyze_person3, compute_acoustic_score, compute_final_score, validate_gemini_response


class FakeResponse:
    text = json.dumps({"emotion": "frustrated", "agitation_score": 0.75, "behaviours": ["complaining", "repetitive question"], "reasoning": "Repeated complaints are present."})


class FakeModels:
    def generate_content(self, **kwargs):
        return FakeResponse()


class FakeClient:
    models = FakeModels()


def test_person3_validates_maps_and_fuses_scores():
    result = analyze_person3("Why are you ignoring me?", {"rms_energy": 0.2}, 0.5, GeminiBehaviourAnalyzer(client=FakeClient()))
    assert result["gemini"]["emotion"] == "frustrated"
    assert result["final_score"] == 0.65
    assert result["cmai_mapping"][0]["cmai_category"].startswith("Verbally non-aggressive")


def test_missing_acoustic_score_is_derived_from_features():
    score = compute_acoustic_score({"rms_energy": 0.3, "pitch_variance": 2500, "speech_rate_wpm": 220, "distress_events": ["shouting"]})
    assert score == 1.0
    result = analyze_person3("help", {"rms_energy": 0.3, "pitch_variance": 2500}, analyzer=GeminiBehaviourAnalyzer(client=FakeClient()))
    assert result["acoustic_score"] == 1.0
    assert result["final_score"] == 0.85


def test_invalid_gemini_score_is_rejected():
    with pytest.raises(ValueError):
        validate_gemini_response({"emotion": "calm", "agitation_score": 1.1, "behaviours": [], "reasoning": ""})

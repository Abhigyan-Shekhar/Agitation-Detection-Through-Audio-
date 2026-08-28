from __future__ import annotations

import sys
import types
import pytest

from person2_module import (
    EmbeddingService,
    BehaviourEvidenceResult,
    HashingTextEmbeddingProvider,
    Person2Config,
    SentenceTransformerEmbeddingProvider,
    analyze_person1_transcript,
    contextual_chunk_transcript,
    cosine_similarity,
    detect_semantic_repetition_evidence,
    deduplicate_behaviours,
    detect_repetitions,
    supported_person2_behaviours,
    transcript_only_excluded_behaviours,
)


class CountingEmbeddingProvider:
    model_name = "counting-test-embedding"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text)), 1.0, 0.0]


class FailingEmbeddingProvider:
    model_name = "failing-test-embedding"

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("model unavailable")


class SemanticTestEmbeddingProvider:
    model_name = "semantic-test-embedding"

    def embed(self, text: str) -> list[float]:
        lower = text.lower()
        if "daughter" in lower:
            return [1.0, 0.0, 0.0, 0.0]
        if "food" in lower or "eat" in lower:
            return [0.0, 1.0, 0.0, 0.0]
        if "weather" in lower:
            return [0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0]


class PrototypeTestEmbeddingProvider:
    model_name = "prototype-test-embedding"

    def embed(self, text: str) -> list[float]:
        lower = text.lower()
        if "medicine" in lower or "refus" in lower or "comply" in lower:
            return [1.0, 0.0, 0.0, 0.0, 0.0]
        if "help" in lower or "attention" in lower:
            return [0.0, 1.0, 0.0, 0.0, 0.0]
        if "terrible" in lower or "complain" in lower:
            return [0.0, 0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0, 1.0, 0.0]


class FakeMiniLMModel:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        assert normalize_embeddings is True
        return [[1.0] + [0.0] * 383 for _ in texts]


def _person1_segments():
    return [
        {"start": 0.0, "end": 2.0, "text": "Where is my daughter?", "confidence": 0.9},
        {"start": 2.5, "end": 4.0, "text": "where is my daughter", "confidence": 0.8},
        {"start": 4.5, "end": 6.0, "text": "Where is my daughter!", "confidence": 0.85},
        {"start": 8.0, "end": 10.0, "text": "The nurse is here.", "confidence": 0.95},
        {"start": 11.0, "end": 13.0, "text": "I won't take my medicine.", "confidence": 0.75},
    ]


def test_default_person2_configuration_uses_20_second_minilm_baseline():
    settings = Person2Config()

    assert settings.max_chunk_duration_sec == 20
    assert settings.max_segments_per_chunk == 8
    assert settings.overlap_segments == 1
    assert settings.embedding_backend == "sentence-transformers"
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.embedding_dimension == 384


def test_contextual_chunking_groups_segments_with_overlap_and_preserves_timestamps():
    settings = Person2Config(max_chunk_duration_sec=6.0, max_segments_per_chunk=3, overlap_segments=1)

    chunks = contextual_chunk_transcript(_person1_segments(), settings)

    assert len(chunks) == 3
    assert chunks[0].start == 0.0
    assert chunks[0].end == 6.0
    assert chunks[0].segment_indices == [0, 1, 2]
    assert chunks[1].segment_indices[0] == 2
    assert chunks[1].start == 4.5
    assert chunks[1].end == 10.0
    assert chunks[2].start == 8.0
    assert chunks[2].end == 13.0


def test_repetition_detection_normalizes_case_punctuation_and_keeps_occurrence_timestamps():
    settings = Person2Config(max_chunk_duration_sec=10.0, max_segments_per_chunk=5, repetition_min_occurrences=3)
    chunk = contextual_chunk_transcript(_person1_segments()[:3], settings)[0]

    repetitions = detect_repetitions(chunk, settings)

    assert len(repetitions) == 1
    evidence = repetitions[0]
    assert evidence.normalized_phrase == "where is my daughter"
    assert evidence.count == 3
    assert evidence.start == 0.0
    assert evidence.end == 6.0
    assert evidence.is_question is True
    assert [item.start for item in evidence.occurrences] == [0.0, 2.5, 4.5]


def test_repetition_detection_ignores_non_repeated_text():
    settings = Person2Config(max_chunk_duration_sec=10.0, max_segments_per_chunk=5)
    chunk = contextual_chunk_transcript(
        [
            {"start": 0.0, "end": 1.0, "text": "Good morning."},
            {"start": 2.0, "end": 3.0, "text": "The tea is warm."},
        ],
        settings,
    )[0]

    assert detect_repetitions(chunk, settings) == []


def test_20_second_context_window_keeps_local_repetition_without_unrelated_later_speech():
    chunks = contextual_chunk_transcript(
        [
            {"start": 10.0, "end": 11.0, "text": "Where is my daughter?"},
            {"start": 14.0, "end": 15.0, "text": "Where is my daughter?"},
            {"start": 18.0, "end": 19.0, "text": "Where is my daughter?"},
            {"start": 35.0, "end": 36.0, "text": "I want some water."},
            {"start": 42.0, "end": 43.0, "text": "The television is too loud."},
        ]
    )

    assert chunks[0].start == 10.0
    assert chunks[0].end == 19.0
    assert [segment.text for segment in chunks[0].segments] == [
        "Where is my daughter?",
        "Where is my daughter?",
        "Where is my daughter?",
    ]
    assert all(chunk.end - chunk.start <= 20.0 for chunk in chunks)


def test_hashing_embedding_provider_returns_stable_fixed_size_vector():
    provider = HashingTextEmbeddingProvider(dimension=8, model_name="test-hashing")

    first = provider.embed("Where is my daughter?")
    second = provider.embed("Where is my daughter?")

    assert first == second
    assert len(first) == 8
    assert all(isinstance(value, float) for value in first)


def test_sentence_transformer_provider_reuses_loaded_model_and_returns_384_dimensions():
    model = FakeMiniLMModel()
    SentenceTransformerEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2", model=model)
    provider = SentenceTransformerEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2")

    first = provider.embed("Where is my daughter?")
    second = provider.embed("The weather is nice today.")

    assert len(first) == 384
    assert len(second) == 384
    assert model.calls == 2


def test_sentence_transformer_constructor_runs_once_across_providers(monkeypatch):
    model_name = "test/reusable-minilm"
    created = []

    class FakeSentenceTransformer:
        def __init__(self, name):
            created.append(name)

        def encode(self, texts, normalize_embeddings=True):
            return [[1.0, 0.0] for _ in texts]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    SentenceTransformerEmbeddingProvider._MODEL_CACHE.pop(model_name, None)

    first = SentenceTransformerEmbeddingProvider(model_name)
    second = SentenceTransformerEmbeddingProvider(model_name)
    assert first.embed("first") == [1.0, 0.0]
    assert second.embed("second") == [1.0, 0.0]
    assert created == [model_name]


def test_embedding_service_caches_duplicate_chunk_text():
    settings = Person2Config(max_chunk_duration_sec=5.0, max_segments_per_chunk=1, overlap_segments=0)
    chunks = contextual_chunk_transcript(
        [
            {"start": 0.0, "end": 1.0, "text": "Repeat me."},
            {"start": 2.0, "end": 3.0, "text": "Repeat me."},
        ],
        settings,
    )
    provider = CountingEmbeddingProvider()
    service = EmbeddingService(provider)

    embedded = [service.embed_chunk(chunk) for chunk in chunks]

    assert provider.calls == 1
    assert embedded[0].embedding == embedded[1].embedding
    assert embedded[0].embedding_model == "counting-test-embedding"


def test_embedding_failure_is_reported_without_crashing_pipeline():
    chunk = contextual_chunk_transcript([{"start": 0.0, "end": 1.0, "text": "Hello."}])[0]
    embedded = EmbeddingService(FailingEmbeddingProvider()).embed_chunk(chunk)

    assert embedded.embedding == []
    assert "model unavailable" in str(embedded.embedding_error)


def test_semantic_similarity_related_question_is_higher_than_unrelated_text():
    provider = SemanticTestEmbeddingProvider()

    daughter_a = provider.embed("Where is my daughter?")
    daughter_b = provider.embed("I don't know where my daughter is.")
    weather = provider.embed("The weather is nice today.")

    assert cosine_similarity(daughter_a, daughter_b) > cosine_similarity(daughter_a, weather)


def test_semantic_repetition_evidence_supports_wording_variation_for_questions():
    settings = Person2Config(semantic_similarity_threshold=0.70)
    chunk = contextual_chunk_transcript(
        [
            {"start": 0.0, "end": 2.0, "text": "Where is my daughter?"},
            {"start": 3.0, "end": 5.0, "text": "I don't know where my daughter is."},
        ],
        settings,
    )[0]

    evidence = detect_semantic_repetition_evidence(chunk, SemanticTestEmbeddingProvider(), settings)

    assert len(evidence) == 1
    assert evidence[0].is_question is True
    assert evidence[0].semantic_matches[0].similarity >= 0.70
    assert evidence[0].start == 0.0
    assert evidence[0].end == 5.0


def test_semantic_similarity_does_not_relabel_complaints_as_repetitive_questions():
    result = analyze_person1_transcript(
        [
            {"start": 0.0, "end": 2.0, "text": "I hate this food."},
            {"start": 3.0, "end": 5.0, "text": "I don't like this food."},
            {"start": 6.0, "end": 8.0, "text": "This food is terrible."},
        ],
        embedding_provider=SemanticTestEmbeddingProvider(),
    )

    labels = {item.behaviour for item in result.behaviours}
    assert "Repetitive sentences or questions" not in labels


def test_person2_detects_repetitive_questioning_with_canonical_taxonomy_label():
    result = analyze_person1_transcript(
        _person1_segments()[:3],
        settings=Person2Config(max_chunk_duration_sec=10.0, max_segments_per_chunk=5),
        embedding_provider=CountingEmbeddingProvider(),
    )

    contract = result.behaviour_contract()
    assert contract
    repeated = [
        item for item in contract
        if item["behaviour"] == "Repetitive sentences or questions"
        and item["score_type"] == "heuristic_repetition_score"
    ]
    assert repeated
    assert repeated[0]["internal_code"] == "AUDIO_REPETITIVE"
    assert repeated[0]["start"] == 0.0
    assert repeated[0]["end"] == 6.0
    assert repeated[0]["score_type"] == "heuristic_repetition_score"
    assert repeated[0]["repetition"]["count"] == 3


def test_person2_output_includes_semantic_similarity_as_supporting_evidence():
    result = analyze_person1_transcript(
        [
            {"start": 0.0, "end": 2.0, "text": "Where is my daughter?"},
            {"start": 3.0, "end": 5.0, "text": "I don't know where my daughter is."},
        ],
        embedding_provider=SemanticTestEmbeddingProvider(),
    )

    semantic_results = [
        item for item in result.behaviour_contract()
        if item["score_type"] == "semantic_similarity_supported_repetition_score"
    ]
    assert semantic_results
    assert semantic_results[0]["behaviour"] == "Repetitive sentences or questions"
    assert semantic_results[0]["repetition"]["semantic_matches"]


def test_person2_detects_other_transcript_supported_cmai_behaviours():
    result = analyze_person1_transcript(
        [
            {"start": 0.0, "end": 2.0, "text": "What the fuck is going on?"},
            {"start": 3.0, "end": 5.0, "text": "I'm tired of this."},
            {"start": 6.0, "end": 8.0, "text": "I won't take my medicine."},
            {"start": 9.0, "end": 11.0, "text": "[moaning]"},
            {"start": 12.0, "end": 14.0, "text": "Please help me now."},
        ],
        settings=Person2Config(max_chunk_duration_sec=3.0, max_segments_per_chunk=1, overlap_segments=0),
        embedding_provider=CountingEmbeddingProvider(),
    )

    labels = {item.behaviour for item in result.behaviours}
    assert "Cursing / verbal aggression" in labels
    assert "Complaining" in labels
    assert "Negativism" in labels
    assert "Making strange noises" in labels
    assert "Distressed/urgent verbalization" in labels


def test_person2_linguistic_behaviour_uses_evidence_segment_not_context_span():
    result = analyze_person1_transcript(
        [
            {"id": "seg-a", "start": 0.0, "end": 2.0, "text": "The weather is fine."},
            {"id": "seg-b", "start": 10.0, "end": 11.2, "text": "I won't take my medicine."},
        ],
        settings=Person2Config(max_chunk_duration_sec=20.0, max_segments_per_chunk=4),
        embedding_provider=CountingEmbeddingProvider(),
    )

    negativism = [item for item in result.behaviours if item.behaviour == "Negativism"]

    assert negativism
    assert negativism[0].start == 10.0
    assert negativism[0].end == 11.2
    assert negativism[0].source_segment_ids == ["seg-b"]
    assert negativism[0].context_start == 0.0
    assert negativism[0].context_end == 11.2


def test_semantic_prototypes_nominate_negativism_without_regex_match():
    result = analyze_person1_transcript(
        [{"id": "seg-1", "start": 5.0, "end": 6.0, "text": "Refusing medicine is happening here."}],
        settings=Person2Config(prototype_similarity_threshold=0.90),
        embedding_provider=PrototypeTestEmbeddingProvider(),
    )

    candidates = [item for item in result.behaviours if item.score_type == "semantic_prototype_candidate_score"]
    assert any(item.behaviour == "Negativism" and item.source_segment_ids == ["seg-1"] for item in candidates)


def test_person2_deduplicates_overlap_chunks_without_chunk_id_identity():
    first = BehaviourEvidenceResult(
        start=10.0,
        end=12.0,
        behaviour="Negativism",
        internal_code="AUDIO_NEGATIVISM",
        cmai_category="Verbally non-aggressive: negativism/refusal",
        score=0.6,
        score_type="heuristic_linguistic_score",
        evidence="first",
        text="I won't go.",
        chunk_id="chunk-0001",
        source_segment_ids=["seg-9"],
        evidence_segments=[{"id": "seg-9", "start": 10.0, "end": 12.0, "text": "I won't go."}],
    )
    second = BehaviourEvidenceResult(
        start=10.1,
        end=12.1,
        behaviour="Negativism",
        internal_code="AUDIO_NEGATIVISM",
        cmai_category="Verbally non-aggressive: negativism/refusal",
        score=0.9,
        score_type="semantic_prototype_candidate_score",
        evidence="second",
        text="I won't go.",
        chunk_id="chunk-0002",
        source_segment_ids=["seg-9"],
        evidence_segments=[{"id": "seg-9", "start": 10.1, "end": 12.1, "text": "I won't go."}],
    )

    merged = deduplicate_behaviours([first, second], Person2Config())

    assert len(merged) == 1
    assert merged[0].score == 0.9
    assert merged[0].chunk_id == "chunk-0002"


def test_person2_does_not_claim_transcript_only_screaming_or_physical_behaviours():
    result = analyze_person1_transcript(
        [
            {"start": 0.0, "end": 2.0, "text": "Stop yelling at me."},
            {"start": 3.0, "end": 5.0, "text": "The resident is pacing."},
        ],
        settings=Person2Config(max_chunk_duration_sec=3.0, max_segments_per_chunk=1, overlap_segments=0),
        embedding_provider=CountingEmbeddingProvider(),
    )

    labels = {item.behaviour for item in result.behaviours}
    assert "Screaming" not in labels
    assert "Pacing" not in labels


def test_supported_and_excluded_person2_behaviour_lists_are_taxonomy_aligned():
    supported = supported_person2_behaviours()
    excluded = transcript_only_excluded_behaviours()

    assert "Repetitive sentences or questions" in supported
    assert "Constant unwarranted requests for attention/help" in supported
    assert "Screaming" in excluded


def test_person1_to_person2_to_person3_output_contract_contains_required_fields():
    result = analyze_person1_transcript(
        _person1_segments(),
        settings=Person2Config(max_chunk_duration_sec=10.0, max_segments_per_chunk=5),
        embedding_provider=CountingEmbeddingProvider(),
    )

    assert result.chunks
    assert result.embedded_chunks
    assert result.behaviours
    for item in result.behaviour_contract():
        assert {
            "start",
            "end",
            "behaviour",
            "internal_code",
            "cmai_category",
            "score",
            "score_type",
            "evidence",
            "text",
            "chunk_id",
            "modality",
            "mapping_status",
        }.issubset(item)
        assert item["start"] <= item["end"]
        assert item["mapping_status"] == "mapped"
        assert item["modality"] == "audio"


def test_invalid_person1_timestamps_are_rejected():
    with pytest.raises(ValueError, match="ordered timestamps"):
        contextual_chunk_transcript([{"start": 5.0, "end": 4.0, "text": "bad"}])


def test_strong_acoustic_agitation_flags_neutral_words_but_loud_constant_speech_does_not():
    result = analyze_person1_transcript(
        [
            {"start": 0.0, "end": 2.0, "text": "Fine. Whatever.", "confidence": 0.95,
             "acoustic": {"available": True, "agitation_score": 0.88, "scream_score": 0.20,
                          "relative_energy": 4.0, "burst_score": 0.70}},
            {"start": 3.0, "end": 5.0, "text": "Welcome everyone.", "confidence": 0.95,
             "acoustic": {"available": True, "agitation_score": 0.20, "scream_score": 0.20,
                          "relative_energy": 1.0, "burst_score": 0.02}},
        ], settings=Person2Config(max_chunk_duration_sec=3.0, max_segments_per_chunk=1, overlap_segments=0),
        embedding_provider=CountingEmbeddingProvider(),
    )
    assert any(item.behaviour == "Vocal agitation" and item.start == 0.0 for item in result.behaviours)
    assert not any(item.behaviour == "Vocal agitation" and item.start == 3.0 for item in result.behaviours)


def test_acoustic_only_extreme_scream_reaches_person2_when_asr_has_no_text():
    result = analyze_person1_transcript(
        [{"start": 10.0, "end": 10.5, "text": "", "acoustic": {
            "available": True, "agitation_score": 0.90, "scream_score": 0.96,
            "relative_energy": 5.0, "burst_score": 0.85,
        }}],
        embedding_provider=CountingEmbeddingProvider(),
    )
    assert any(item.behaviour == "Screaming" for item in result.behaviours)


def test_legacy_transcript_record_without_acoustics_still_works():
    result = analyze_person1_transcript(
        [{"start": 0.0, "end": 1.0, "text": "Please help me now.", "confidence": 0.95}],
        embedding_provider=CountingEmbeddingProvider(),
    )
    assert any(item.behaviour == "Distressed/urgent verbalization" for item in result.behaviours)

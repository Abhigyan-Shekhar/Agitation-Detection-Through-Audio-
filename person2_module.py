"""Person 2 batch transcript evidence layer.

This module consumes Person 1's timestamped transcript contract and emits
timestamp-preserving, CMAI-aligned initial behaviour evidence for Person 3.
It does not run ASR, alter upload handling, or call any LLM service.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
from typing import Any, Protocol

import config
from audio_behaviour_taxonomy import get_supported_behaviours, map_observed_behaviour
from event_models import CommittedLine, Utterance
from linguistic_features import LinguisticAnalyzer, _fuzzy_similarity, _is_question, _is_request, _normalize


@dataclass(frozen=True)
class Person1TranscriptSegment:
    """Person 1 transcript segment consumed by this module."""

    start: float
    end: float
    text: str
    confidence: float | None = None
    acoustic: dict[str, Any] | None = None


@dataclass(frozen=True)
class Person2Config:
    """Tunable parameters for contextual chunking, repetition, and embeddings."""

    max_chunk_duration_sec: float = config.PERSON2_CHUNK_MAX_DURATION_SEC
    max_segments_per_chunk: int = config.PERSON2_CHUNK_MAX_SEGMENTS
    overlap_segments: int = config.PERSON2_CHUNK_OVERLAP_SEGMENTS
    repetition_min_occurrences: int = config.PERSON2_REPETITION_MIN_OCCURRENCES
    repetition_similarity_threshold: float = config.PERSON2_REPETITION_SIMILARITY_THRESHOLD
    semantic_similarity_threshold: float = config.PERSON2_SEMANTIC_SIMILARITY_THRESHOLD
    embedding_backend: str = config.PERSON2_EMBEDDING_BACKEND
    embedding_model: str = config.PERSON2_EMBEDDING_MODEL
    embedding_dimension: int = config.PERSON2_EMBEDDING_DIMENSION


@dataclass(frozen=True)
class TranscriptChunk:
    """Contextual unit with timestamp links back to original segments."""

    chunk_id: str
    start: float
    end: float
    text: str
    segments: list[Person1TranscriptSegment]
    segment_indices: list[int]


@dataclass(frozen=True)
class RepetitionOccurrence:
    """One occurrence of a repeated phrase."""

    start: float
    end: float
    text: str
    segment_index: int


@dataclass(frozen=True)
class SemanticSimilarityEvidence:
    """Semantic relationship between nearby utterances in one chunk."""

    first_text: str
    second_text: str
    first_start: float
    first_end: float
    second_start: float
    second_end: float
    similarity: float
    is_question_pair: bool
    is_request_pair: bool


@dataclass(frozen=True)
class RepetitionEvidence:
    """Explainable repetition evidence found inside one chunk."""

    repeated_phrase: str
    normalized_phrase: str
    count: int
    start: float
    end: float
    occurrences: list[RepetitionOccurrence]
    is_question: bool
    is_request: bool
    surrounding_text: str
    score: float
    semantic_matches: list[SemanticSimilarityEvidence] = field(default_factory=list)


@dataclass(frozen=True)
class EmbeddedChunk:
    """Chunk plus embedding metadata."""

    chunk: TranscriptChunk
    embedding: list[float]
    embedding_model: str
    embedding_error: str | None = None


@dataclass(frozen=True)
class BehaviourEvidenceResult:
    """Person 3-ready initial behaviour signal."""

    start: float
    end: float
    behaviour: str
    internal_code: str
    cmai_category: str
    score: float
    score_type: str
    evidence: str
    text: str
    chunk_id: str
    repetition: dict[str, Any] | None = None
    modality: str = "audio"
    mapping_status: str = "mapped"
    acoustic: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Person2AnalysisResult:
    """Full Person 2 output with internals retained for audit/tests."""

    chunks: list[TranscriptChunk]
    repetitions: dict[str, list[RepetitionEvidence]]
    embedded_chunks: list[EmbeddedChunk]
    behaviours: list[BehaviourEvidenceResult]

    def behaviour_contract(self) -> list[dict[str, Any]]:
        """Return the stable structured evidence payload for Person 3."""
        return [behaviour.as_dict() for behaviour in self.behaviours]


class TextEmbeddingProvider(Protocol):
    """Small embedding interface so tests and future model backends can plug in."""

    model_name: str

    def embed(self, text: str) -> list[float]:
        """Return one fixed-size embedding vector for text."""


class HashingTextEmbeddingProvider:
    """Deterministic, dependency-free text embedding provider.

    The vector is a normalized signed hashing projection over word tokens. It
    is lightweight enough for local tests and batch evidence generation, but it
    is a semantic feature vector rather than a clinical classifier.
    """

    def __init__(
        self,
        *,
        dimension: int = config.PERSON2_EMBEDDING_DIMENSION,
        model_name: str = config.PERSON2_EMBEDDING_MODEL,
    ) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dimension = dimension
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9']+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [round(value / norm, 8) for value in vector]


class SentenceTransformerEmbeddingProvider:
    """Optional sentence-transformers backend, loaded only when configured."""

    _MODEL_CACHE: dict[str, Any] = {}

    def __init__(self, model_name: str, model: Any | None = None) -> None:
        self.model_name = model_name
        if model is not None:
            self._MODEL_CACHE[model_name] = model

    @property
    def _model(self) -> Any:
        model = self._MODEL_CACHE.get(self.model_name)
        if model is not None:
            return model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("sentence-transformers is not installed") from exc
        model = SentenceTransformer(self.model_name)
        self._MODEL_CACHE[self.model_name] = model
        return model

    def embed(self, text: str) -> list[float]:
        embedding = self._model.encode([text], normalize_embeddings=True)[0]
        return [float(value) for value in embedding]


class EmbeddingService:
    """Cached embedding generation for contextual chunks."""

    def __init__(self, provider: TextEmbeddingProvider) -> None:
        self.provider = provider
        self._cache: dict[str, list[float]] = {}

    def embed_chunk(self, chunk: TranscriptChunk) -> EmbeddedChunk:
        try:
            if chunk.text not in self._cache:
                self._cache[chunk.text] = self.provider.embed(chunk.text)
            return EmbeddedChunk(chunk=chunk, embedding=list(self._cache[chunk.text]), embedding_model=self.provider.model_name)
        except Exception as exc:  # noqa: BLE001
            return EmbeddedChunk(chunk=chunk, embedding=[], embedding_model=self.provider.model_name, embedding_error=str(exc))


def build_default_embedding_provider(settings: Person2Config | None = None) -> TextEmbeddingProvider:
    """Build the configured embedding provider without forcing heavy dependencies."""
    settings = settings or Person2Config()
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(settings.embedding_model)
    if settings.embedding_backend != "hashing":
        raise ValueError(f"Unsupported Person 2 embedding backend: {settings.embedding_backend!r}")
    return HashingTextEmbeddingProvider(dimension=settings.embedding_dimension, model_name=settings.embedding_model)


def coerce_person1_transcript(raw_segments: list[Person1TranscriptSegment | dict[str, Any] | Any]) -> list[Person1TranscriptSegment]:
    """Validate and normalize Person 1 transcript records without changing timestamps."""
    segments: list[Person1TranscriptSegment] = []
    for item in raw_segments:
        if isinstance(item, Person1TranscriptSegment):
            segment = item
        elif isinstance(item, dict):
            segment = Person1TranscriptSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]).strip(),
                confidence=_coerce_confidence(item.get("confidence")),
                acoustic=_coerce_acoustic(item.get("acoustic")),
            )
        else:
            segment = Person1TranscriptSegment(
                start=float(getattr(item, "start")),
                end=float(getattr(item, "end")),
                text=str(getattr(item, "text")).strip(),
                confidence=_coerce_confidence(getattr(item, "confidence", None)),
                acoustic=_coerce_acoustic(getattr(item, "acoustic", None)),
            )
        if segment.start < 0 or segment.end < segment.start:
            raise ValueError("Person 1 transcript segments must have non-negative ordered timestamps")
        if segment.text or segment.acoustic is not None:
            segments.append(segment)
    return segments


def contextual_chunk_transcript(
    raw_segments: list[Person1TranscriptSegment | dict[str, Any] | Any],
    settings: Person2Config | None = None,
) -> list[TranscriptChunk]:
    """Group adjacent transcript segments into overlapping contextual chunks."""
    settings = settings or Person2Config()
    if settings.max_chunk_duration_sec <= 0 or settings.max_segments_per_chunk <= 0:
        raise ValueError("Chunk duration and segment limits must be positive")
    if settings.overlap_segments < 0:
        raise ValueError("Chunk overlap cannot be negative")

    segments = coerce_person1_transcript(raw_segments)
    chunks: list[TranscriptChunk] = []
    cursor = 0
    while cursor < len(segments):
        selected: list[Person1TranscriptSegment] = []
        selected_indices: list[int] = []
        chunk_start = segments[cursor].start
        idx = cursor
        while idx < len(segments) and len(selected) < settings.max_segments_per_chunk:
            candidate = segments[idx]
            if selected and candidate.end - chunk_start > settings.max_chunk_duration_sec:
                break
            selected.append(candidate)
            selected_indices.append(idx)
            idx += 1

        chunk = TranscriptChunk(
            chunk_id=f"chunk-{len(chunks):04d}",
            start=min(segment.start for segment in selected),
            end=max(segment.end for segment in selected),
            text=" ".join(segment.text for segment in selected if segment.text).strip(),
            segments=selected,
            segment_indices=selected_indices,
        )
        chunks.append(chunk)
        if idx >= len(segments):
            break
        next_cursor = idx - min(settings.overlap_segments, max(0, len(selected) - 1))
        cursor = max(cursor + 1, next_cursor)
    return chunks


def detect_repetitions(chunk: TranscriptChunk, settings: Person2Config | None = None) -> list[RepetitionEvidence]:
    """Find repeated nearby phrases inside a contextual chunk."""
    settings = settings or Person2Config()
    groups: list[list[RepetitionOccurrence]] = []
    representatives: list[str] = []

    for local_idx, segment in enumerate(chunk.segments):
        normalized = _normalize(segment.text)
        if not normalized:
            continue
        occurrence = RepetitionOccurrence(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            segment_index=chunk.segment_indices[local_idx],
        )
        matched_group = None
        for idx, representative in enumerate(representatives):
            if normalized == representative or _fuzzy_similarity(normalized, representative) >= settings.repetition_similarity_threshold:
                matched_group = idx
                break
        if matched_group is None:
            representatives.append(normalized)
            groups.append([occurrence])
        else:
            groups[matched_group].append(occurrence)

    evidence: list[RepetitionEvidence] = []
    for representative, occurrences in zip(representatives, groups, strict=True):
        if len(occurrences) < settings.repetition_min_occurrences:
            continue
        start = min(item.start for item in occurrences)
        end = max(item.end for item in occurrences)
        count = len(occurrences)
        score = min(1.0, 0.55 + 0.15 * count)
        repeated_phrase = occurrences[0].text
        evidence.append(
            RepetitionEvidence(
                repeated_phrase=repeated_phrase,
                normalized_phrase=representative,
                count=count,
                start=start,
                end=end,
                occurrences=occurrences,
                is_question=any(_is_question(item.text) for item in occurrences),
                is_request=any(_is_request(item.text) for item in occurrences),
                surrounding_text=chunk.text,
                score=round(score, 3),
            )
        )
    return evidence


def detect_semantic_repetition_evidence(
    chunk: TranscriptChunk,
    provider: TextEmbeddingProvider,
    settings: Person2Config | None = None,
) -> list[RepetitionEvidence]:
    """Use semantic embeddings as supporting evidence for local repetition.

    Semantic similarity can support repetitive sentence/question evidence only
    when the pair is question-like or request-like. Similar complaint-like
    statements are left to the complaint/negativism heuristics instead of being
    relabelled as repetitive sentences/questions.
    """
    settings = settings or Person2Config()
    if len(chunk.segments) < settings.repetition_min_occurrences:
        return []

    embedded: list[tuple[Person1TranscriptSegment, list[float]]] = []
    for segment in chunk.segments:
        try:
            embedding = provider.embed(segment.text)
        except Exception:  # noqa: BLE001
            return []
        if embedding:
            embedded.append((segment, embedding))

    evidence: list[RepetitionEvidence] = []
    seen_pairs: set[tuple[float, float, float, float]] = set()
    for idx, (left, left_embedding) in enumerate(embedded):
        for right, right_embedding in embedded[idx + 1 :]:
            similarity = cosine_similarity(left_embedding, right_embedding)
            if similarity < settings.semantic_similarity_threshold:
                continue
            is_question_pair = _is_question(left.text) or _is_question(right.text)
            is_request_pair = _is_request(left.text) or _is_request(right.text)
            if not (is_question_pair or is_request_pair):
                continue
            pair_key = (left.start, left.end, right.start, right.end)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            semantic_match = SemanticSimilarityEvidence(
                first_text=left.text,
                second_text=right.text,
                first_start=left.start,
                first_end=left.end,
                second_start=right.start,
                second_end=right.end,
                similarity=round(similarity, 3),
                is_question_pair=is_question_pair,
                is_request_pair=is_request_pair,
            )
            score = min(1.0, 0.50 + 0.35 * similarity)
            evidence.append(
                RepetitionEvidence(
                    repeated_phrase=left.text,
                    normalized_phrase=_normalize(left.text),
                    count=2,
                    start=min(left.start, right.start),
                    end=max(left.end, right.end),
                    occurrences=[
                        RepetitionOccurrence(left.start, left.end, left.text, chunk.segment_indices[chunk.segments.index(left)]),
                        RepetitionOccurrence(right.start, right.end, right.text, chunk.segment_indices[chunk.segments.index(right)]),
                    ],
                    is_question=is_question_pair,
                    is_request=is_request_pair,
                    surrounding_text=chunk.text,
                    score=round(score, 3),
                    semantic_matches=[semantic_match],
                )
            )
    return evidence


def embed_chunks(
    chunks: list[TranscriptChunk],
    provider: TextEmbeddingProvider | None = None,
    settings: Person2Config | None = None,
) -> list[EmbeddedChunk]:
    """Generate cached embeddings for contextual chunks."""
    service = EmbeddingService(provider or build_default_embedding_provider(settings))
    return [service.embed_chunk(chunk) for chunk in chunks]


def detect_initial_behaviours(
    chunks: list[TranscriptChunk],
    repetitions: dict[str, list[RepetitionEvidence]],
) -> list[BehaviourEvidenceResult]:
    """Emit initial, transcript-supported CMAI-aligned behaviour evidence."""
    analyzer = LinguisticAnalyzer(history_sec=config.TRANSCRIPT_HISTORY_SEC)
    results: list[BehaviourEvidenceResult] = []
    seen: set[tuple[str, str, float, float]] = set()

    for chunk in chunks:
        utterance = _chunk_to_utterance(chunk)
        features = analyzer.analyze(utterance)
        chunk_repetitions = repetitions.get(chunk.chunk_id, [])

        for repetition in chunk_repetitions:
            label = "Repeated requests" if repetition.is_request else "Repeated questioning" if repetition.is_question else "Repetitive verbalization"
            mapped = map_observed_behaviour(label)
            if mapped.mapping_status == "mapped":
                score_type = (
                    "semantic_similarity_supported_repetition_score"
                    if repetition.semantic_matches
                    else "heuristic_repetition_score"
                )
                evidence_text = (
                    f"Semantically similar question/request pair found "
                    f"(similarity={repetition.semantic_matches[0].similarity:.3f}); "
                    "treated as supporting evidence, not a CMAI probability."
                    if repetition.semantic_matches
                    else f"Phrase repeated {repetition.count} times within nearby transcript segments."
                )
                results.append(
                    BehaviourEvidenceResult(
                        start=repetition.start,
                        end=repetition.end,
                        behaviour=mapped.canonical_label,
                        internal_code=str(mapped.internal_code),
                        cmai_category=mapped.cmai_category,
                        score=repetition.score,
                        score_type=score_type,
                        evidence=evidence_text,
                        text=repetition.surrounding_text,
                        chunk_id=chunk.chunk_id,
                        repetition={
                            "repeated_phrase": repetition.repeated_phrase,
                            "count": repetition.count,
                            "occurrences": [asdict(item) for item in repetition.occurrences],
                            "is_question": repetition.is_question,
                            "is_request": repetition.is_request,
                            "semantic_matches": [asdict(item) for item in repetition.semantic_matches],
                        },
                    )
                )

        linguistic_candidates = [
            ("Cursing / verbal aggression", features.profanity_score, "Explicit profanity/verbal aggression cue in transcript."),
            ("Making verbal sexual advances", features.sexual_advance_score, "Sexualized verbal proposition or comment in transcript."),
            ("Complaining", features.complaint_score, "Complaint semantics detected in transcript."),
            ("Negativism", features.negativism_score, "Refusal/resistance/non-compliance/defiance language detected in transcript."),
            ("Making strange noises", features.strange_noise_score, "Transcript contains a non-speech vocalization label."),
            ("Distressed/urgent verbalization", features.urgency_score, "Urgent help-seeking, escape, or stop cue in transcript."),
        ]
        for label, score, evidence in linguistic_candidates:
            threshold = _threshold_for(label)
            if score < threshold:
                continue
            mapped = map_observed_behaviour(label)
            if mapped.mapping_status != "mapped":
                continue
            result = BehaviourEvidenceResult(
                start=chunk.start,
                end=chunk.end,
                behaviour=mapped.canonical_label,
                internal_code=str(mapped.internal_code),
                cmai_category=mapped.cmai_category,
                score=round(float(score), 3),
                score_type="heuristic_linguistic_score",
                evidence=evidence,
                text=chunk.text,
                chunk_id=chunk.chunk_id,
            )
            key = (result.behaviour, result.chunk_id, result.start, result.end)
            if key not in seen:
                seen.add(key)
                results.append(result)

        # Acoustic evidence is aligned by Person 1 to each original segment,
        # not inferred by this text layer.  Keep these candidates segment
        # scoped so a loud utterance does not contaminate a whole 20-second
        # contextual chunk.
        for segment in chunk.segments:
            acoustic = segment.acoustic or {}
            if not acoustic.get("available"):
                continue
            agitation = float(acoustic.get("agitation_score", 0.0))
            scream = float(acoustic.get("scream_score", 0.0))
            urgency = features.urgency_score if segment.text else 0.0
            if scream >= config.PERSON2_ACOUSTIC_SCREAM_THRESHOLD:
                label, score, rationale = "Screaming/shouting", scream, "Strong timestamp-aligned scream evidence from source audio."
            elif agitation >= config.PERSON2_ACOUSTIC_AGITATION_THRESHOLD:
                label, score, rationale = "Vocal agitation", agitation, "Strong timestamp-aligned acoustic agitation with no requirement for hostile wording."
            elif agitation >= config.PERSON2_ACOUSTIC_COMBINED_THRESHOLD and urgency >= config.BEHAVIOUR_URGENCY_THRESHOLD:
                label, score, rationale = "Distressed/urgent verbalization", min(1.0, (agitation + urgency) / 2), "Urgent language corroborated by acoustic activation."
            else:
                continue
            mapped = map_observed_behaviour(label)
            if mapped.mapping_status != "mapped":
                continue
            result = BehaviourEvidenceResult(
                start=segment.start, end=segment.end, behaviour=mapped.canonical_label,
                internal_code=str(mapped.internal_code), cmai_category=mapped.cmai_category,
                score=round(score, 3), score_type="fused_acoustic_linguistic_score",
                evidence=(f"{rationale} acoustic_agitation={agitation:.2f}, scream_score={scream:.2f}, "
                          f"relative_energy={float(acoustic.get('relative_energy', 0.0)):.2f}, "
                          f"burst={float(acoustic.get('burst_score', 0.0)):.2f}, urgency={urgency:.2f}"),
                text=segment.text, chunk_id=chunk.chunk_id, acoustic=acoustic,
            )
            key = (result.behaviour, result.chunk_id, result.start, result.end)
            if key not in seen:
                seen.add(key)
                results.append(result)

    return sorted(results, key=lambda item: (item.start, item.end, item.behaviour))


def analyze_person1_transcript(
    raw_segments: list[Person1TranscriptSegment | dict[str, Any] | Any],
    *,
    settings: Person2Config | None = None,
    embedding_provider: TextEmbeddingProvider | None = None,
) -> Person2AnalysisResult:
    """Run the full Person 2 pipeline from Person 1 transcript to evidence."""
    settings = settings or Person2Config()
    chunks = contextual_chunk_transcript(raw_segments, settings)
    provider = embedding_provider or build_default_embedding_provider(settings)
    repetitions = {
        chunk.chunk_id: (
            detect_repetitions(chunk, settings)
            + detect_semantic_repetition_evidence(chunk, provider, settings)
        )
        for chunk in chunks
    }
    embedded = embed_chunks(chunks, provider=provider, settings=settings)
    behaviours = detect_initial_behaviours(chunks, repetitions)
    return Person2AnalysisResult(
        chunks=chunks,
        repetitions=repetitions,
        embedded_chunks=embedded,
        behaviours=behaviours,
    )


def supported_person2_behaviours() -> dict[str, str]:
    """Return behaviours Person 2 may emit from transcript and upload acoustics."""
    detectable_codes = {
        "AUDIO_CURSING",
        "AUDIO_VERBAL_SEXUAL_ADVANCES",
        "AUDIO_REPETITIVE",
        "AUDIO_STRANGE_NOISE",
        "AUDIO_COMPLAINING",
        "AUDIO_NEGATIVISM",
        "AUDIO_CONSTANT_REQUEST",
        "AUDIO_URGENT_DISTRESS",
        "AUDIO_VOCAL_AGITATION",
        "AUDIO_SCREAMING",
    }
    return {
        entry.canonical_label: entry.description
        for entry in get_supported_behaviours()
        if entry.internal_code in detectable_codes
    }


def transcript_only_excluded_behaviours() -> dict[str, str]:
    """Describe labels unavailable to legacy records that omit acoustic data."""
    return {
        entry.canonical_label: "Requires timestamp-aligned acoustic evidence; legacy transcript-only records remain supported."
        for entry in get_supported_behaviours()
        if entry.internal_code in {"AUDIO_SCREAMING", "AUDIO_VOCAL_AGITATION"}
    }


def _chunk_to_utterance(chunk: TranscriptChunk) -> Utterance:
    lines = [
        CommittedLine(
            text=segment.text,
            timestamp=segment.end,
            start_time=segment.start,
            end_time=segment.end,
            transcript_confidence=segment.confidence,
        )
        for segment in chunk.segments
    ]
    return Utterance(lines=lines, start_time=chunk.start, end_time=chunk.end)


def _threshold_for(label: str) -> float:
    if label == "Cursing / verbal aggression":
        return 0.50
    if label == "Making verbal sexual advances":
        return 0.60
    if label == "Complaining":
        return config.BEHAVIOUR_COMPLAINT_THRESHOLD
    if label == "Negativism":
        return config.BEHAVIOUR_NEGATIVISM_THRESHOLD
    if label == "Making strange noises":
        return config.BEHAVIOUR_STRANGE_NOISE_THRESHOLD
    if label == "Distressed/urgent verbalization":
        return config.BEHAVIOUR_URGENCY_THRESHOLD
    return 1.0


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    return min(1.0, max(0.0, float(value)))


def _coerce_acoustic(value: Any) -> dict[str, Any] | None:
    """Accept additive Person 1 metadata without making legacy records fail."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Person 1 acoustic evidence must be an object when supplied")
    evidence = dict(value)
    for key in ("agitation_score", "scream_score", "relative_energy", "burst_score", "rms_mean", "rms_peak", "clipping_ratio", "voiced_ratio"):
        if key in evidence:
            evidence[key] = float(evidence[key])
    return evidence


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))

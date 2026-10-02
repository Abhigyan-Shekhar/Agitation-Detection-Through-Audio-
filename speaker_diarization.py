"""Local, online speaker embedding and session-scoped clustering.

This module consumes audio already captured for faster-whisper. It does not
open a microphone, use WhisperLiveKit, or send audio to a cloud service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Protocol

import numpy as np

import config

logger = logging.getLogger(__name__)


class SpeakerEmbeddingBackend(Protocol):
    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray: ...


class SpeechBrainECAPABackend:
    """Lazy SpeechBrain ECAPA-TDNN embedding adapter."""

    def __init__(self, model_source: str = config.DIARIZATION_MODEL) -> None:
        self._model_source = model_source
        self._classifier = None

    def _load(self):
        if self._classifier is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError as exc:
                raise RuntimeError(
                    "Speaker diarization requires SpeechBrain. Install "
                    "requirements-diarization.txt or set ENABLE_SPEAKER_DIARIZATION=false."
                ) from exc
            logger.info("Loading local speaker embedding model: %s", self._model_source)
            self._classifier = EncoderClassifier.from_hparams(source=self._model_source)
        return self._classifier

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 16_000:
            raise ValueError("SpeechBrain ECAPA diarization expects 16 kHz audio")
        import torch

        waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            embedding = self._load().encode_batch(waveform)
        return embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)


@dataclass(frozen=True)
class PatientSpeakerMatch:
    """Binary upload-time match against the enrolled patient voiceprint."""

    label: str
    is_patient: bool
    similarity: float


class EnrolledPatientSpeakerIdentifier:
    """Enroll the first patient-audio minute and verify later speech against it.

    The enrollment embedding is held in memory for the lifetime of this object.
    It is deliberately not written to disk: callers can decide whether their
    consent and retention policy permits persistent biometric storage.
    """

    def __init__(
        self,
        backend: SpeakerEmbeddingBackend | None = None,
        *,
        similarity_threshold: float = config.DIARIZATION_SIMILARITY_THRESHOLD,
        min_segment_seconds: float = config.DIARIZATION_MIN_SEGMENT_SECONDS,
    ) -> None:
        self._backend = backend or SpeechBrainECAPABackend()
        self._threshold = float(similarity_threshold)
        self._min_seconds = float(min_segment_seconds)
        self._patient_embedding: np.ndarray | None = None

    @property
    def enrolled(self) -> bool:
        return self._patient_embedding is not None

    @property
    def enrollment_embedding(self) -> np.ndarray | None:
        """Return a defensive copy for an explicitly authorized persistence layer."""
        if self._patient_embedding is None:
            return None
        return self._patient_embedding.copy()

    def enroll(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size < int(self._min_seconds * sample_rate):
            raise ValueError("Patient enrollment audio is too short for speaker embedding")
        vector = self._normalise(self._backend.embed(samples, sample_rate))
        if not np.any(vector):
            raise ValueError("Patient enrollment produced an empty speaker embedding")
        self._patient_embedding = vector
        return vector.copy()

    def identify(self, audio: np.ndarray, sample_rate: int) -> PatientSpeakerMatch | None:
        if self._patient_embedding is None:
            raise RuntimeError("Patient speaker has not been enrolled")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size < int(self._min_seconds * sample_rate):
            return None
        vector = self._normalise(self._backend.embed(samples, sample_rate))
        if not np.any(vector):
            return None
        similarity = float(np.dot(vector, self._patient_embedding))
        is_patient = similarity >= self._threshold
        return PatientSpeakerMatch(
            label="Patient" if is_patient else "Other speaker",
            is_patient=is_patient,
            similarity=similarity,
        )

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else np.zeros_like(vector)


@dataclass
class _SpeakerCentroid:
    vector: np.ndarray
    observations: int = 1
    # A centroid alone can blur different speaking styles from one person.
    # Retaining a small prototype bank makes short/noisy turns match any
    # previously observed mode without allowing memory to grow unbounded.
    prototypes: list[np.ndarray] = field(default_factory=list)


class OnlineSpeakerDiarizer:
    """Assign embeddings to stable Speaker 1..N labels for one session."""

    def __init__(
        self,
        backend: SpeakerEmbeddingBackend | None = None,
        similarity_threshold: float = config.DIARIZATION_SIMILARITY_THRESHOLD,
        min_segment_seconds: float = config.DIARIZATION_MIN_SEGMENT_SECONDS,
        max_speakers: int = config.DIARIZATION_MAX_SPEAKERS,
    ) -> None:
        if max_speakers < 1:
            raise ValueError("max_speakers must be at least 1")
        self._backend = backend or SpeechBrainECAPABackend()
        self._threshold = similarity_threshold
        self._min_seconds = min_segment_seconds
        self._max_speakers = max_speakers
        self._speakers: dict[int, _SpeakerCentroid] = {}

    @property
    def speakers_seen(self) -> int:
        return len(self._speakers)

    def reset(self) -> None:
        self._speakers.clear()

    def identify(self, audio: np.ndarray, sample_rate: int) -> tuple[int | None, str | None]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size < int(self._min_seconds * sample_rate):
            return None, None
        vector = self._normalise(self._backend.embed(samples, sample_rate))
        if not np.any(vector):
            return None, None

        speaker_id: int
        if not self._speakers:
            speaker_id = 1
            self._speakers[speaker_id] = _SpeakerCentroid(vector, prototypes=[vector])
        else:
            similarities = {
                sid: max(
                    float(np.dot(vector, centroid.vector)),
                    *(float(np.dot(vector, prototype)) for prototype in centroid.prototypes),
                )
                for sid, centroid in self._speakers.items()
            }
            speaker_id, best = max(similarities.items(), key=lambda item: item[1])
            logger.info(
                "DIARIZATION_MATCH speaker=%s similarity=%.3f threshold=%.3f",
                speaker_id,
                best,
                self._threshold,
            )
            if best < self._threshold and len(self._speakers) < self._max_speakers:
                speaker_id = max(self._speakers) + 1
                self._speakers[speaker_id] = _SpeakerCentroid(vector, prototypes=[vector])
                logger.info("DIARIZATION speakers_seen=%d", len(self._speakers))
            else:
                centroid = self._speakers[speaker_id]
                centroid.vector = self._normalise(
                    centroid.vector * centroid.observations + vector
                )
                centroid.observations += 1
                centroid.prototypes.append(vector)
                if len(centroid.prototypes) > 12:
                    centroid.prototypes.pop(0)
        return speaker_id, f"Speaker {speaker_id}"

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else np.zeros_like(vector)

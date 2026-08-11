"""Local, online speaker embedding and session-scoped clustering.

This module consumes audio already captured for faster-whisper. It does not
open a microphone, use WhisperLiveKit, or send audio to a cloud service.
"""
from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class _SpeakerCentroid:
    vector: np.ndarray
    observations: int = 1


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
            self._speakers[speaker_id] = _SpeakerCentroid(vector)
        else:
            similarities = {
                sid: float(np.dot(vector, centroid.vector))
                for sid, centroid in self._speakers.items()
            }
            speaker_id, best = max(similarities.items(), key=lambda item: item[1])
            if best < self._threshold and len(self._speakers) < self._max_speakers:
                speaker_id = max(self._speakers) + 1
                self._speakers[speaker_id] = _SpeakerCentroid(vector)
                logger.info("DIARIZATION speakers_seen=%d", len(self._speakers))
            else:
                centroid = self._speakers[speaker_id]
                centroid.vector = self._normalise(
                    centroid.vector * centroid.observations + vector
                )
                centroid.observations += 1
        return speaker_id, f"Speaker {speaker_id}"

    @staticmethod
    def _normalise(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-8 else np.zeros_like(vector)

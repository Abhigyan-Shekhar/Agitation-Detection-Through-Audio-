"""Batch audio ingestion and timestamped transcription for Person 1.

The public ``transcribe_upload`` function is the integration boundary for the
next pipeline stage. It validates an uploaded file, decodes it to mono 16 kHz
float PCM, and returns transcript segments using audio-relative timestamps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np

import config
from transcriber import DirectWhisperTranscriber, TranscriptSegment


SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".webm"})
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 2 * 60 * 60


class AudioValidationError(ValueError):
    """Raised when an upload cannot safely be processed as supported audio."""


@dataclass(frozen=True)
class ProcessedAudio:
    """Decoded audio ready for local ASR."""

    samples: np.ndarray
    sample_rate: int
    duration: float
    filename: str
    source_bytes: int


@dataclass(frozen=True)
class TimestampedTranscript:
    """Stable Person 1 -> Person 2 transcript contract."""

    start: float
    end: float
    text: str
    confidence: float | None = None

    def as_dict(self) -> dict[str, str | float | None]:
        return asdict(self)


@dataclass(frozen=True)
class BatchTranscriptionResult:
    """Timestamped transcript plus non-contract processing metadata."""

    segments: list[TimestampedTranscript]
    filename: str
    duration: float
    sample_rate: int
    model: str

    def transcript_contract(self) -> list[dict[str, str | float | None]]:
        """Return the JSON-ready payload consumed by Person 2."""
        return [segment.as_dict() for segment in self.segments]


def validate_upload(filename: str, data: bytes, *, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    """Perform cheap checks before opening an untrusted upload with FFmpeg."""
    if not filename or Path(filename).suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise AudioValidationError(f"Unsupported audio type. Allowed extensions: {allowed}")
    if not data:
        raise AudioValidationError("The uploaded audio file is empty.")
    if len(data) > max_bytes:
        raise AudioValidationError(f"Audio file exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")


def preprocess_upload(
    data: bytes,
    filename: str,
    *,
    target_sample_rate: int = config.SAMPLE_RATE,
    max_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
) -> ProcessedAudio:
    """Decode uploaded audio and resample it to finite mono float32 PCM.

    Leading/trailing silence is intentionally retained. Removing it would make
    transcript offsets disagree with playback offsets in the original file.
    """
    validate_upload(filename, data)
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    chunks: list[np.ndarray] = []
    try:
        with av.open(BytesIO(data), mode="r") as container:
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if not audio_streams:
                raise AudioValidationError("The uploaded file does not contain an audio stream.")
            stream = audio_streams[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sample_rate)
            for frame in container.decode(stream):
                _append_resampled(chunks, resampler.resample(frame))
            _append_resampled(chunks, resampler.resample(None))
    except AudioValidationError:
        raise
    except (av.error.FFmpegError, EOFError, OSError, ValueError) as exc:
        raise AudioValidationError("The file could not be decoded as valid audio.") from exc

    if not chunks:
        raise AudioValidationError("No decodable audio samples were found in the file.")
    samples = np.concatenate(chunks).astype(np.float32, copy=False)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    samples = np.clip(samples, -1.0, 1.0)
    duration = float(samples.size / target_sample_rate)
    if duration <= 0:
        raise AudioValidationError("The audio has zero duration.")
    if duration > max_duration_seconds:
        limit_minutes = max_duration_seconds / 60
        raise AudioValidationError(f"Audio duration exceeds the {limit_minutes:g}-minute processing limit.")
    return ProcessedAudio(samples, target_sample_rate, duration, Path(filename).name, len(data))


def _append_resampled(chunks: list[np.ndarray], frames: Any) -> None:
    if frames is None:
        return
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    for frame in frames:
        array = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
        if array.size:
            chunks.append(array)


def transcribe_upload(
    data: bytes,
    filename: str,
    *,
    transcriber: DirectWhisperTranscriber | Any | None = None,
) -> BatchTranscriptionResult:
    """Process one uploaded file into the Person 1 timestamped contract."""
    processed = preprocess_upload(data, filename)
    engine = transcriber or DirectWhisperTranscriber(sample_rate=processed.sample_rate)
    _text, raw_segments, _confidence = engine.transcribe(processed.samples)
    segments = list(_normalise_segments(raw_segments, processed.duration))
    model_name = str(getattr(engine, "model_size", config.WHISPER_MODEL))
    return BatchTranscriptionResult(
        segments=segments,
        filename=processed.filename,
        duration=processed.duration,
        sample_rate=processed.sample_rate,
        model=model_name,
    )


def _normalise_segments(
    segments: Iterable[TranscriptSegment | Any], duration: float
) -> Iterable[TimestampedTranscript]:
    previous_end = 0.0
    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        raw_start = getattr(segment, "start", previous_end)
        raw_end = getattr(segment, "end", raw_start)
        start = min(duration, max(0.0, float(previous_end if raw_start is None else raw_start)))
        end = min(duration, max(start, float(start if raw_end is None else raw_end)))
        confidence = getattr(segment, "confidence", None)
        if confidence is not None:
            confidence = min(1.0, max(0.0, float(confidence)))
        yield TimestampedTranscript(
            start=round(start, 3),
            end=round(end, 3),
            text=text,
            confidence=confidence,
        )
        previous_end = end

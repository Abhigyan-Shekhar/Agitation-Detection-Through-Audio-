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
    acoustic: dict[str, float | bool] | None = None

    def as_dict(self, *, include_acoustic: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_acoustic:
            payload.pop("acoustic", None)
        return payload


@dataclass(frozen=True)
class BatchTranscriptionResult:
    """Timestamped transcript plus non-contract processing metadata."""

    segments: list[TimestampedTranscript]
    filename: str
    duration: float
    sample_rate: int
    model: str
    acoustic_events: list[TimestampedTranscript] | None = None

    def transcript_contract(self) -> list[dict[str, Any]]:
        """Return the JSON-ready payload consumed by Person 2."""
        # Preserve the public download/UI payload used by existing callers.
        # The richer additive record is exposed through person2_contract().
        return [segment.as_dict(include_acoustic=False) for segment in self.segments]

    def person2_contract(self) -> list[dict[str, Any]]:
        """Return transcript records plus acoustic-only upload events.

        Empty-text event records intentionally remain out of the public
        transcript contract; they exist so a scream missed by ASR can still
        reach Person 2.
        """
        return [segment.as_dict() for segment in self.segments] + [event.as_dict() for event in self.acoustic_events or []]


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
    baseline_rms = _recording_baseline_rms(processed.samples, processed.sample_rate)
    segments = [_with_acoustic(segment, processed.samples, processed.sample_rate, baseline_rms) for segment in segments]
    acoustic_events = _acoustic_only_events(processed.samples, processed.sample_rate, baseline_rms)
    model_name = str(getattr(engine, "model_size", config.WHISPER_MODEL))
    return BatchTranscriptionResult(
        segments=segments,
        filename=processed.filename,
        duration=processed.duration,
        sample_rate=processed.sample_rate,
        model=model_name,
        acoustic_events=acoustic_events,
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


def _recording_baseline_rms(samples: np.ndarray, sample_rate: int) -> float:
    """Robust recording-relative energy baseline, excluding silence."""
    frame = max(1, int(sample_rate * 0.05))
    values = [float(np.sqrt(np.mean(samples[i:i + frame] ** 2))) for i in range(0, samples.size, frame) if samples[i:i + frame].size]
    active = [value for value in values if value >= 0.003]
    return float(np.median(active)) if active else 0.003


def _with_acoustic(segment: TimestampedTranscript, samples: np.ndarray, sample_rate: int, baseline_rms: float) -> TimestampedTranscript:
    start = max(0, int(segment.start * sample_rate))
    end = min(samples.size, max(start + 1, int(segment.end * sample_rate)))
    return TimestampedTranscript(segment.start, segment.end, segment.text, segment.confidence,
                                 _acoustic_evidence(samples[start:end], sample_rate, baseline_rms))


def _acoustic_evidence(audio: np.ndarray, sample_rate: int, baseline_rms: float) -> dict[str, float | bool]:
    """Small, deterministic upload feature set aligned to one time region.

    These are recording-relative features, deliberately requiring more than
    loudness: a constant loud microphone gain has little burst or relative
    deviation and therefore does not become vocal agitation by itself.
    """
    y = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
    if y.size == 0:
        return {"available": False, "agitation_score": 0.0, "scream_score": 0.0}
    frame = max(1, int(sample_rate * 0.04))
    rms = np.asarray([np.sqrt(np.mean(y[i:i + frame] ** 2)) for i in range(0, y.size, frame)], dtype=float)
    mean_rms = float(np.mean(rms))
    peak = float(np.max(np.abs(y)))
    relative = mean_rms / max(baseline_rms, 1e-4)
    burst = float(max(0.0, (np.percentile(rms, 95) - np.median(rms)) / max(np.percentile(rms, 95), 1e-4)))
    clipping = float(np.mean(np.abs(y) >= 0.99))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(y))))) if y.size > 1 else 0.0
    # Speech-like periodic signals generally have a moderate crossing rate;
    # this prevents isolated impacts/very high-frequency noise from carrying
    # the same weight as a voiced yell.
    voice_like = 0.01 <= zcr <= 0.30
    voiced_ratio = float(np.mean(rms >= max(0.008, baseline_rms * 0.30)))
    relative_score = float(np.clip((relative - 1.5) / 3.0, 0.0, 1.0))
    energy_score = float(np.clip((mean_rms - 0.10) / 0.25, 0.0, 1.0))
    peak_score = float(np.clip((peak - 0.55) / 0.40, 0.0, 1.0))
    agitation = float(np.clip(0.42 * relative_score + 0.30 * burst + 0.18 * energy_score + 0.10 * min(1.0, clipping / 0.04), 0.0, 1.0))
    scream = float(np.clip((0.38 * energy_score + 0.30 * peak_score + 0.18 * burst + 0.14 * min(1.0, clipping / 0.04)) * (1.0 if voice_like and voiced_ratio >= 0.30 else 0.35), 0.0, 1.0))
    return {
        "available": True, "rms_mean": round(mean_rms, 4), "rms_peak": round(peak, 4),
        "relative_energy": round(relative, 3), "clipping_ratio": round(clipping, 4),
        "voiced_ratio": round(voiced_ratio, 3), "burst_score": round(burst, 3),
        "voice_like": voice_like, "agitation_score": round(agitation, 3), "scream_score": round(scream, 3),
    }


def _acoustic_only_events(samples: np.ndarray, sample_rate: int, baseline_rms: float) -> list[TimestampedTranscript]:
    """Emit merged fixed-window scream events even when Whisper has no text."""
    window, hop = 1.0, 0.5
    positives: list[tuple[float, float, dict[str, float | bool]]] = []
    for start in np.arange(0.0, samples.size / sample_rate, hop):
        end = min(samples.size / sample_rate, start + window)
        evidence = _acoustic_evidence(samples[int(start * sample_rate):int(end * sample_rate)], sample_rate, baseline_rms)
        score = float(evidence["scream_score"])
        if score >= config.SCREAM_EXTREME_SCORE_THRESHOLD:
            positives.append((float(start), float(end), evidence))
        elif score >= config.PERSON2_ACOUSTIC_SCREAM_THRESHOLD:
            positives.append((float(start), float(end), evidence))
    events: list[TimestampedTranscript] = []
    for start, end, evidence in positives:
        if events and start <= events[-1].end:
            previous = events[-1]
            events[-1] = TimestampedTranscript(previous.start, max(previous.end, end), "", None, evidence)
        else:
            events.append(TimestampedTranscript(start, end, "", None, evidence))
    # Borderline candidates need the configured number of overlapping windows.
    # With 1.0 s windows and 0.5 s hops, three positives cover 2.0 seconds;
    # checking only SCREAM_MIN_DURATION_SEC would accidentally accept one
    # window because its analysis window itself is already one second long.
    required_coverage = max(
        config.SCREAM_MIN_DURATION_SEC,
        window + (config.SCREAM_MIN_CONSECUTIVE_WINDOWS - 1) * hop,
    )
    # Extreme candidates are allowed immediately and are marked in their
    # evidence; ordinary candidates must meet the real window-count coverage.
    return [event for event in events if float(event.acoustic["scream_score"]) >= config.SCREAM_EXTREME_SCORE_THRESHOLD or event.end - event.start >= required_coverage]

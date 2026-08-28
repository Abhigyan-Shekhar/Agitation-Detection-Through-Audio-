"""Batch audio ingestion and timestamped transcription for Person 1.

The public ``transcribe_upload`` function is the integration boundary for the
next pipeline stage. It validates an uploaded file, decodes it to mono 16 kHz
float PCM, and returns transcript segments using audio-relative timestamps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import collections
from io import BytesIO
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Iterable, Iterator

import av
import numpy as np

import config
from transcriber import DirectWhisperTranscriber, TranscriptSegment, TranscriptWord


SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg", ".oga", ".webm"})
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 2 * 60 * 60
LOGGER = logging.getLogger(__name__)
_BATCH_TRANSCRIBER_CACHE: dict[tuple[Any, ...], DirectWhisperTranscriber] = {}
_BATCH_TRANSCRIBER_CACHE_LOCK = threading.Lock()


class AudioValidationError(ValueError):
    """Raised when an upload cannot safely be processed as supported audio."""


class BatchTranscriberLoadError(RuntimeError):
    """Raised when the configured upload Whisper model cannot be initialized."""


@dataclass(frozen=True)
class ProcessedAudio:
    """Decoded audio ready for local ASR."""

    samples: np.ndarray
    sample_rate: int
    duration: float
    filename: str
    source_bytes: int


@dataclass(frozen=True)
class AudioMetadata:
    """Cheap upload metadata that does not require materialising decoded PCM."""

    filename: str
    duration: float
    sample_rate: int | None
    source_bytes: int


@dataclass(frozen=True)
class TranscriptionAudioChunk:
    """One bounded ASR input window with absolute source-audio timing."""

    samples: np.ndarray
    sample_rate: int
    input_start: float
    primary_start: float
    primary_end: float
    duration: float
    index: int
    total: int


@dataclass(frozen=True)
class TimestampedTranscript:
    """Stable Person 1 -> Person 2 transcript contract."""

    start: float
    end: float
    text: str
    confidence: float | None = None
    acoustic: dict[str, float | bool] | None = None
    id: str | None = None
    source_segment_ids: list[str] | None = None

    def as_dict(self, *, include_acoustic: bool = True, include_source_metadata: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_acoustic:
            payload.pop("acoustic", None)
        if not include_source_metadata:
            payload.pop("id", None)
            payload.pop("source_segment_ids", None)
        elif payload.get("source_segment_ids") is None and payload.get("id"):
            payload["source_segment_ids"] = [payload["id"]]
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
        return [segment.as_dict(include_acoustic=False, include_source_metadata=False) for segment in self.segments]

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


def inspect_upload(
    data: bytes,
    filename: str,
    *,
    target_sample_rate: int = config.SAMPLE_RATE,
    max_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
) -> AudioMetadata:
    """Return validated upload metadata without decoding the whole recording."""
    validate_upload(filename, data)
    try:
        with av.open(BytesIO(data), mode="r") as container:
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if not audio_streams:
                raise AudioValidationError("The uploaded file does not contain an audio stream.")
            stream = audio_streams[0]
            duration = _stream_duration_seconds(container, stream)
            if duration is None:
                duration = _decoded_duration_seconds(data, target_sample_rate)
    except AudioValidationError:
        raise
    except (av.error.FFmpegError, EOFError, OSError, ValueError) as exc:
        raise AudioValidationError("The file could not be decoded as valid audio.") from exc

    if duration <= 0:
        raise AudioValidationError("The audio has zero duration.")
    if duration > max_duration_seconds:
        limit_minutes = max_duration_seconds / 60
        raise AudioValidationError(f"Audio duration exceeds the {limit_minutes:g}-minute processing limit.")
    return AudioMetadata(Path(filename).name, duration, getattr(stream, "rate", None), len(data))


def _stream_duration_seconds(container: av.container.InputContainer, stream: av.audio.stream.AudioStream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    return None


def _decoded_duration_seconds(data: bytes, target_sample_rate: int) -> float:
    total_samples = 0
    for samples in _iter_resampled_arrays(data, target_sample_rate):
        total_samples += int(samples.size)
    return float(total_samples / target_sample_rate)


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


def iter_transcription_chunks(
    data: bytes,
    filename: str,
    *,
    target_sample_rate: int = config.SAMPLE_RATE,
    chunk_seconds: float = config.BATCH_TRANSCRIPTION_CHUNK_SECONDS,
    overlap_seconds: float = config.BATCH_TRANSCRIPTION_OVERLAP_SECONDS,
    max_duration_seconds: float = MAX_AUDIO_DURATION_SECONDS,
    metadata: AudioMetadata | None = None,
) -> Iterator[TranscriptionAudioChunk]:
    """Yield bounded ASR chunks that cover the full upload exactly once.

    Each yielded ``samples`` array may include leading overlap for model
    context. The ``primary_*`` interval is the non-overlapping source-audio
    range owned by that chunk.
    """
    metadata = metadata or inspect_upload(
        data, filename, target_sample_rate=target_sample_rate, max_duration_seconds=max_duration_seconds,
    )
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds cannot be negative")
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    chunk_samples = max(1, int(round(chunk_seconds * target_sample_rate)))
    overlap_samples = max(0, int(round(overlap_seconds * target_sample_rate)))
    total_samples = max(1, int(round(metadata.duration * target_sample_rate)))
    total_chunks = max(1, int(np.ceil(total_samples / chunk_samples)))
    carry = np.empty(0, dtype=np.float32)
    pending: list[np.ndarray] = []
    pending_samples = 0
    primary_start_sample = 0
    index = 0

    for array in _iter_resampled_arrays(data, target_sample_rate):
        pending.append(array)
        pending_samples += int(array.size)
        while pending_samples >= chunk_samples:
            primary, pending_samples = _take_samples(pending, chunk_samples)
            input_samples = np.concatenate([carry, primary]) if carry.size else primary
            input_start_sample = max(0, primary_start_sample - carry.size)
            primary_end_sample = primary_start_sample + primary.size
            yield TranscriptionAudioChunk(
                samples=np.ascontiguousarray(input_samples, dtype=np.float32),
                sample_rate=target_sample_rate,
                input_start=input_start_sample / target_sample_rate,
                primary_start=primary_start_sample / target_sample_rate,
                primary_end=primary_end_sample / target_sample_rate,
                duration=metadata.duration,
                index=index,
                total=total_chunks,
            )
            carry = input_samples[-overlap_samples:] if overlap_samples else np.empty(0, dtype=np.float32)
            primary_start_sample = primary_end_sample
            index += 1

    if pending_samples or index == 0:
        primary, pending_samples = _take_samples(pending, pending_samples)
        input_samples = np.concatenate([carry, primary]) if carry.size else primary
        input_start_sample = max(0, primary_start_sample - carry.size)
        primary_end_sample = primary_start_sample + primary.size
        yield TranscriptionAudioChunk(
            samples=np.ascontiguousarray(input_samples, dtype=np.float32),
            sample_rate=target_sample_rate,
            input_start=input_start_sample / target_sample_rate,
            primary_start=primary_start_sample / target_sample_rate,
            primary_end=primary_end_sample / target_sample_rate,
            duration=metadata.duration,
            index=index,
            total=total_chunks,
        )


def _take_samples(pending: list[np.ndarray], count: int) -> tuple[np.ndarray, int]:
    """Pop exactly ``count`` samples from pending decoder arrays."""
    if count <= 0:
        return np.empty(0, dtype=np.float32), sum(array.size for array in pending)
    parts: list[np.ndarray] = []
    remaining = count
    while pending and remaining > 0:
        head = pending.pop(0)
        if head.size <= remaining:
            parts.append(head)
            remaining -= int(head.size)
        else:
            parts.append(head[:remaining])
            pending.insert(0, head[remaining:])
            remaining = 0
    samples = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(samples, dtype=np.float32), sum(array.size for array in pending)


def _iter_resampled_arrays(data: bytes, target_sample_rate: int) -> Iterator[np.ndarray]:
    try:
        with av.open(BytesIO(data), mode="r") as container:
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if not audio_streams:
                raise AudioValidationError("The uploaded file does not contain an audio stream.")
            stream = audio_streams[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sample_rate)
            for frame in container.decode(stream):
                for array in _resampled_arrays(resampler.resample(frame)):
                    yield array
            for array in _resampled_arrays(resampler.resample(None)):
                yield array
    except AudioValidationError:
        raise
    except (av.error.FFmpegError, EOFError, OSError, ValueError) as exc:
        raise AudioValidationError("The file could not be decoded as valid audio.") from exc


def _resampled_arrays(frames: Any) -> Iterator[np.ndarray]:
    if frames is None:
        return
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    for frame in frames:
        array = np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1)
        if array.size:
            clean = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=-1.0)
            yield np.clip(clean, -1.0, 1.0).astype(np.float32, copy=False)


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
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    chunk_seconds: float = config.BATCH_TRANSCRIPTION_CHUNK_SECONDS,
    overlap_seconds: float = config.BATCH_TRANSCRIPTION_OVERLAP_SECONDS,
) -> BatchTranscriptionResult:
    """Process one uploaded file into the Person 1 timestamped contract."""
    started = time.monotonic()
    if progress_callback is not None:
        progress_callback("preparing", 0, 1, "Validating audio...")
    inspection_started = time.monotonic()
    metadata = inspect_upload(data, filename)
    LOGGER.info("Audio inspection complete filename=%s duration=%.2fs elapsed=%.2fs", metadata.filename, metadata.duration, time.monotonic() - inspection_started)
    sample_rate = config.SAMPLE_RATE
    engine = transcriber or get_batch_transcriber(sample_rate=sample_rate, progress_callback=progress_callback)
    if progress_callback is not None:
        progress_callback("preparing", 1, 1, "Preparing transcription...")
    segments: list[TimestampedTranscript] = []
    acoustic_events: list[TimestampedTranscript] = []
    baseline_history: collections.deque[float] = collections.deque(maxlen=60)
    transcription_started = time.monotonic()
    chunks_processed = 0
    for chunk in iter_transcription_chunks(
        data,
        filename,
        target_sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        metadata=metadata,
    ):
        chunks_processed += 1
        chunk_started = time.monotonic()
        chunk_baseline = _recording_baseline_rms(chunk.samples, chunk.sample_rate)
        baseline_history.append(chunk_baseline)
        baseline_rms = float(np.median(baseline_history))
        if progress_callback is not None:
            progress_callback(
                "transcribing",
                chunk.index,
                chunk.total,
                f"Transcribing chunk {chunk.index + 1}/{chunk.total} ({chunk.primary_start:.1f}s-{chunk.primary_end:.1f}s)...",
            )
        _text, raw_segments, _confidence = engine.transcribe(chunk.samples)
        for segment in _normalise_segments(
            raw_segments,
            chunk.duration,
            offset=chunk.input_start,
            include_start=chunk.primary_start,
            include_end=chunk.primary_end,
        ):
            segments.append(_with_acoustic(segment, chunk.samples, chunk.sample_rate, baseline_rms, audio_start=chunk.input_start))
        _append_chunk_acoustic_events(acoustic_events, chunk, baseline_rms)
        chunk_elapsed = time.monotonic() - chunk_started
        LOGGER.info("Transcription chunk %d/%d complete source=%.1f-%.1fs elapsed=%.2fs", chunk.index + 1, chunk.total, chunk.primary_start, chunk.primary_end, chunk_elapsed)
        if progress_callback is not None:
            progress_callback(
                "transcribing",
                chunk.index + 1,
                chunk.total,
                f"Transcribed chunk {chunk.index + 1}/{chunk.total}",
            )
    LOGGER.info("Upload transcription complete chunks=%d elapsed=%.2fs", chunks_processed, time.monotonic() - transcription_started)
    segments.sort(key=lambda item: (item.start, item.end, item.text))
    segments = _assign_segment_ids(segments)
    acoustic_events = _assign_segment_ids(acoustic_events, start_index=len(segments))
    model_name = str(getattr(engine, "model_size", config.WHISPER_MODEL))
    LOGGER.info("Person 1 upload processing complete filename=%s elapsed=%.2fs", metadata.filename, time.monotonic() - started)
    return BatchTranscriptionResult(
        segments=segments,
        filename=metadata.filename,
        duration=metadata.duration,
        sample_rate=sample_rate,
        model=model_name,
        acoustic_events=acoustic_events,
    )


def get_batch_transcriber(
    *,
    sample_rate: int = config.SAMPLE_RATE,
    model_size: str | None = None,
    device: str | None = None,
    compute_type: str | None = None,
    language: str | None = None,
    beam_size: int | None = None,
    word_timestamps: bool | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> DirectWhisperTranscriber:
    """Return one process-wide upload transcriber per compatible configuration."""
    resolved_device = device or ("cuda" if config.USE_GPU_IF_AVAILABLE and DirectWhisperTranscriber._cuda_available() else "cpu")
    resolved_compute = compute_type or ("float16" if resolved_device == "cuda" else "int8")
    resolved_model = model_size or (config.BATCH_WHISPER_GPU_MODEL if resolved_device == "cuda" else config.BATCH_WHISPER_CPU_MODEL)
    resolved_language = config.BATCH_WHISPER_LANGUAGE if language is None else language
    resolved_beam = config.BATCH_WHISPER_BEAM_SIZE if beam_size is None else beam_size
    resolved_words = config.BATCH_WHISPER_WORD_TIMESTAMPS if word_timestamps is None else word_timestamps
    key = (
        resolved_model, resolved_device, resolved_compute, sample_rate, resolved_language,
        resolved_beam, resolved_words, config.BATCH_WHISPER_VAD_MIN_SILENCE_MS,
        config.BATCH_WHISPER_VAD_SPEECH_PAD_MS,
    )
    with _BATCH_TRANSCRIBER_CACHE_LOCK:
        cached = _BATCH_TRANSCRIBER_CACHE.get(key)
        if cached is not None:
            if progress_callback is not None:
                progress_callback("loading_model", 1, 1, f"Reusing Whisper {resolved_model} on {resolved_device.upper()}.")
            return cached
        message = f"Loading Whisper {resolved_model} on {resolved_device.upper()}..."
        if progress_callback is not None:
            progress_callback("loading_model", 0, 1, message)
        load_started = time.monotonic()
        try:
            transcriber = DirectWhisperTranscriber(
                model_size=resolved_model,
                sample_rate=sample_rate,
                language=resolved_language,
                use_gpu_if_available=resolved_device == "cuda",
                beam_size=resolved_beam,
                word_timestamps=resolved_words,
                device=resolved_device,
                compute_type=resolved_compute,
                vad_parameters={
                    "min_silence_duration_ms": config.BATCH_WHISPER_VAD_MIN_SILENCE_MS,
                    "speech_pad_ms": config.BATCH_WHISPER_VAD_SPEECH_PAD_MS,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise BatchTranscriberLoadError(
                f"Failed to load Whisper model {resolved_model!r} on {resolved_device.upper()}: {exc}"
            ) from exc
        _BATCH_TRANSCRIBER_CACHE[key] = transcriber
        elapsed = time.monotonic() - load_started
        LOGGER.info("Whisper model ready model=%s device=%s compute_type=%s elapsed=%.2fs", resolved_model, resolved_device, resolved_compute, elapsed)
        if progress_callback is not None:
            progress_callback("loading_model", 1, 1, f"Whisper {resolved_model} ready in {elapsed:.1f}s.")
        return transcriber


def clear_batch_transcriber_cache() -> None:
    """Clear cached upload transcribers for isolated tests or explicit reconfiguration."""
    with _BATCH_TRANSCRIBER_CACHE_LOCK:
        _BATCH_TRANSCRIBER_CACHE.clear()


def _build_batch_transcriber(sample_rate: int) -> DirectWhisperTranscriber:
    """Backward-compatible wrapper around the cached upload factory."""
    return get_batch_transcriber(sample_rate=sample_rate)


def _append_chunk_acoustic_events(
    events: list[TimestampedTranscript],
    chunk: TranscriptionAudioChunk,
    baseline_rms: float,
) -> None:
    """Reuse decoded ASR PCM for acoustic-only events and merge boundaries."""
    local_events = _acoustic_only_events(chunk.samples, chunk.sample_rate, baseline_rms)
    for event in local_events:
        absolute = TimestampedTranscript(
            round(event.start + chunk.input_start, 3), round(event.end + chunk.input_start, 3), "", None, event.acoustic,
        )
        if absolute.end <= chunk.primary_start or absolute.start >= chunk.primary_end:
            continue
        absolute = replace(absolute, start=max(absolute.start, chunk.primary_start), end=min(absolute.end, chunk.primary_end))
        if events and absolute.start <= events[-1].end:
            previous = events[-1]
            events[-1] = TimestampedTranscript(previous.start, max(previous.end, absolute.end), "", None, absolute.acoustic)
        else:
            events.append(absolute)


def _normalise_segments(
    segments: Iterable[TranscriptSegment | Any],
    duration: float,
    *,
    offset: float = 0.0,
    include_start: float | None = None,
    include_end: float | None = None,
) -> Iterable[TimestampedTranscript]:
    previous_end = offset
    for segment in segments:
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            continue
        words = _normalise_words(getattr(segment, "words", ()), duration, offset=offset)
        if words:
            for unit in _word_units_to_transcripts(words, include_start=include_start, include_end=include_end):
                confidence = _mean_word_confidence(unit)
                yield TimestampedTranscript(
                    start=round(unit[0].start, 3),
                    end=round(unit[-1].end, 3),
                    text=_join_word_text(unit),
                    confidence=confidence,
                )
            previous_end = max(previous_end, max(word.end for word in words))
            continue
        raw_start = getattr(segment, "start", None)
        raw_end = getattr(segment, "end", None)
        start = min(duration, max(0.0, float(previous_end if raw_start is None else raw_start + offset)))
        end = min(duration, max(start, float(start if raw_end is None else raw_end + offset)))
        if include_start is not None and include_end is not None:
            midpoint = start + (end - start) / 2
            if midpoint < include_start or midpoint > include_end:
                previous_end = max(previous_end, end)
                continue
            if offset < include_start and midpoint <= include_start:
                previous_end = max(previous_end, end)
                continue
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


def _normalise_words(
    words: Iterable[TranscriptWord | Any],
    duration: float,
    *,
    offset: float,
) -> list[TranscriptWord]:
    normalized: list[TranscriptWord] = []
    for word in words:
        text = str(getattr(word, "text", getattr(word, "word", ""))).strip()
        raw_start = getattr(word, "start", None)
        raw_end = getattr(word, "end", None)
        if not text or raw_start is None or raw_end is None:
            continue
        start = min(duration, max(0.0, float(raw_start) + offset))
        end = min(duration, max(start, float(raw_end) + offset))
        if end <= start:
            continue
        confidence = getattr(word, "confidence", getattr(word, "probability", None))
        normalized.append(TranscriptWord(text=text, start=start, end=end, confidence=confidence))
    return normalized


def _word_units_to_transcripts(
    words: list[TranscriptWord],
    *,
    include_start: float | None,
    include_end: float | None,
) -> Iterable[list[TranscriptWord]]:
    unit: list[TranscriptWord] = []
    sentence_end_re = re.compile(r"[.!?][\"')\]]*$")
    for word in words:
        if include_start is not None and include_end is not None:
            midpoint = float(word.start or 0.0) + (float(word.end or 0.0) - float(word.start or 0.0)) / 2
            if midpoint < include_start or midpoint > include_end:
                if unit:
                    yield unit
                    unit = []
                continue
        if unit:
            gap = float(word.start or 0.0) - float(unit[-1].end or unit[-1].start or 0.0)
            span = float(word.end or word.start or 0.0) - float(unit[0].start or 0.0)
            if gap >= config.BATCH_TRANSCRIPT_SENTENCE_GAP_SEC or span > config.BATCH_TRANSCRIPT_MAX_UTTERANCE_SEC:
                yield unit
                unit = []
        unit.append(word)
        if sentence_end_re.search(word.text):
            yield unit
            unit = []
    if unit:
        yield unit


def _join_word_text(words: list[TranscriptWord]) -> str:
    text = " ".join(word.text.strip() for word in words if word.text.strip()).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def _mean_word_confidence(words: list[TranscriptWord]) -> float | None:
    values = [float(word.confidence) for word in words if word.confidence is not None]
    if not values:
        return None
    return round(float(np.clip(np.mean(values), 0.0, 1.0)), 4)


def _assign_segment_ids(segments: list[TimestampedTranscript], *, start_index: int = 0) -> list[TimestampedTranscript]:
    assigned: list[TimestampedTranscript] = []
    for index, segment in enumerate(segments, start=start_index):
        segment_id = segment.id or f"seg-{index:06d}"
        assigned.append(replace(segment, id=segment_id, source_segment_ids=segment.source_segment_ids or [segment_id]))
    return assigned


def _recording_baseline_rms(samples: np.ndarray, sample_rate: int) -> float:
    """Robust recording-relative energy baseline, excluding silence."""
    frame = max(1, int(sample_rate * 0.05))
    values = [float(np.sqrt(np.mean(samples[i:i + frame] ** 2))) for i in range(0, samples.size, frame) if samples[i:i + frame].size]
    active = [value for value in values if value >= 0.003]
    return float(np.median(active)) if active else 0.003


def _recording_baseline_rms_streaming(data: bytes, sample_rate: int) -> float:
    """Streaming equivalent of ``_recording_baseline_rms`` for long uploads."""
    frame = max(1, int(sample_rate * 0.05))
    tail = np.empty(0, dtype=np.float32)
    values: list[float] = []
    for chunk in _iter_resampled_arrays(data, sample_rate):
        y = np.concatenate([tail, chunk]) if tail.size else chunk
        full = (y.size // frame) * frame
        if full:
            framed = y[:full].reshape(-1, frame)
            values.extend(float(value) for value in np.sqrt(np.mean(framed ** 2, axis=1)))
        tail = y[full:]
    if tail.size:
        values.append(float(np.sqrt(np.mean(tail ** 2))))
    active = [value for value in values if value >= 0.003]
    return float(np.median(active)) if active else 0.003


def _with_acoustic(
    segment: TimestampedTranscript,
    samples: np.ndarray,
    sample_rate: int,
    baseline_rms: float,
    *,
    audio_start: float = 0.0,
) -> TimestampedTranscript:
    start = max(0, int((segment.start - audio_start) * sample_rate))
    end = min(samples.size, max(start + 1, int((segment.end - audio_start) * sample_rate)))
    local_baseline = _local_baseline_rms(samples, sample_rate, start, end, fallback=baseline_rms)
    return replace(segment, acoustic=_acoustic_evidence(samples[start:end], sample_rate, local_baseline))


def _local_baseline_rms(
    samples: np.ndarray,
    sample_rate: int,
    start_sample: int,
    end_sample: int,
    *,
    fallback: float,
    context_seconds: float = 30.0,
) -> float:
    """Robust local RMS baseline around one evidence span."""
    pad = int(context_seconds * sample_rate)
    start = max(0, start_sample - pad)
    end = min(samples.size, end_sample + pad)
    if end <= start:
        return fallback
    local = _recording_baseline_rms(samples[start:end], sample_rate)
    return float(local if local > 0 else fallback)


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
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        local_baseline = _local_baseline_rms(samples, sample_rate, start_sample, end_sample, fallback=baseline_rms)
        evidence = _acoustic_evidence(samples[start_sample:end_sample], sample_rate, local_baseline)
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


def _acoustic_only_events_streaming(data: bytes, sample_rate: int, baseline_rms: float) -> list[TimestampedTranscript]:
    """Streaming acoustic-only event detection with the same fixed windows."""
    events: list[TimestampedTranscript] = []
    for chunk in iter_transcription_chunks(
        data,
        "recording.wav",
        target_sample_rate=sample_rate,
        chunk_seconds=max(30.0, config.BATCH_TRANSCRIPTION_CHUNK_SECONDS),
        overlap_seconds=1.0,
    ):
        local_events = _acoustic_only_events(chunk.samples, sample_rate, baseline_rms)
        for event in local_events:
            absolute = TimestampedTranscript(
                round(event.start + chunk.input_start, 3),
                round(event.end + chunk.input_start, 3),
                "",
                None,
                event.acoustic,
            )
            if absolute.end <= chunk.primary_start or absolute.start >= chunk.primary_end:
                continue
            absolute = TimestampedTranscript(
                max(absolute.start, chunk.primary_start),
                min(absolute.end, chunk.primary_end),
                "",
                None,
                absolute.acoustic,
            )
            if events and absolute.start <= events[-1].end:
                previous = events[-1]
                events[-1] = TimestampedTranscript(previous.start, max(previous.end, absolute.end), "", None, absolute.acoustic)
            else:
                events.append(absolute)
    return events

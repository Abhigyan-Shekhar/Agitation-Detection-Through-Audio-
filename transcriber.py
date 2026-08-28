"""Local faster-whisper transcription worker for dashboard audio.

The worker consumes ``TimestampedFrame`` objects from ``AudioPipeline`` on a
background thread, maintains a rolling audio window, and emits transcript text
without any external streaming process.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Any, Iterable

import numpy as np

import config
from audio_pipeline import TimestampedFrame
from event_models import CommittedLine, LatencyTrace
from speaker_diarization import OnlineSpeakerDiarizer

logger = logging.getLogger(__name__)

_ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3"}


@dataclass(frozen=True)
class TranscriptWord:
    """One word-level timestamp returned by faster-whisper when enabled."""

    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    """One timestamped segment returned by faster-whisper."""

    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None
    words: tuple[TranscriptWord, ...] = ()


@dataclass(frozen=True)
class TranscriptionResult:
    """A single transcription pass over the current rolling buffer."""

    text: str
    timestamp: float
    segments: list[TranscriptSegment]
    confidence: float | None
    inference_ms: float
    buffer_duration: float


class DirectWhisperTranscriber:
    """Load faster-whisper once and transcribe rolling PCM audio windows."""

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL,
        sample_rate: int = config.SAMPLE_RATE,
        language: str | None = config.WHISPER_LANGUAGE,
        use_gpu_if_available: bool = config.USE_GPU_IF_AVAILABLE,
        beam_size: int = 1,
        word_timestamps: bool = False,
        vad_parameters: dict[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        if model_size not in _ALLOWED_MODELS:
            raise ValueError(f"Unsupported Whisper model {model_size!r}; expected one of {sorted(_ALLOWED_MODELS)}")
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.language = language
        self.use_gpu_if_available = use_gpu_if_available
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps
        self.vad_parameters = vad_parameters
        self.model = model if model is not None else self._load_model()

    def _load_model(self) -> Any:
        logger.info("Loading Whisper model... model=%s", self.model_size)
        from faster_whisper import WhisperModel

        device = "cpu"
        compute_type = "int8"
        if self.use_gpu_if_available and self._cuda_available():
            device = "cuda"
            compute_type = "float16"
        model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
        logger.info("Model loaded. device=%s compute_type=%s", device, compute_type)
        return model

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False

    def transcribe(self, audio: np.ndarray) -> tuple[str, list[TranscriptSegment], float | None]:
        """Return text, segment timestamps, and average confidence if available."""
        if audio.size == 0:
            return "", [], None
        segments_iter, _info = self.model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.language,
            vad_filter=True,
            beam_size=self.beam_size,
            word_timestamps=self.word_timestamps,
            vad_parameters=self.vad_parameters,
        )
        segments = list(self._coerce_segments(segments_iter))
        text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
        confidences = [s.confidence for s in segments if s.confidence is not None]
        confidence = float(np.mean(confidences)) if confidences else None
        return text, segments, confidence

    @staticmethod
    def _coerce_segments(raw_segments: Iterable[Any]) -> Iterable[TranscriptSegment]:
        for segment in raw_segments:
            text = getattr(segment, "text", "")
            start = getattr(segment, "start", None)
            end = getattr(segment, "end", None)
            confidence = getattr(segment, "confidence", None)
            if confidence is None and getattr(segment, "avg_logprob", None) is not None:
                confidence = float(np.exp(float(segment.avg_logprob)))
            words = tuple(_coerce_words(getattr(segment, "words", None)))
            yield TranscriptSegment(text=text, start=start, end=end, confidence=confidence, words=words)


def _coerce_words(raw_words: Iterable[Any] | None) -> Iterable[TranscriptWord]:
    if raw_words is None:
        return
    for word in raw_words:
        text = getattr(word, "word", getattr(word, "text", ""))
        confidence = getattr(word, "probability", getattr(word, "confidence", None))
        yield TranscriptWord(
            text=str(text),
            start=getattr(word, "start", None),
            end=getattr(word, "end", None),
            confidence=float(confidence) if confidence is not None else None,
        )


class TranscriptionWorker:
    """Non-blocking bridge from audio frames to dashboard transcript queues."""

    def __init__(
        self,
        audio_queue: queue.Queue[TimestampedFrame],
        partial_queue: queue.Queue[str],
        committed_queue: queue.Queue[CommittedLine],
        transcriber: DirectWhisperTranscriber | None = None,
        window_seconds: float = config.TRANSCRIPTION_WINDOW_SECONDS,
        interval_seconds: float = config.TRANSCRIPTION_INTERVAL_SECONDS,
        sample_rate: int = config.SAMPLE_RATE,
        diarizer: OnlineSpeakerDiarizer | None = None,
        enable_diarization: bool = config.ENABLE_SPEAKER_DIARIZATION,
    ) -> None:
        self._audio_queue = audio_queue
        self._partial_queue = partial_queue
        self._committed_queue = committed_queue
        self._transcriber = transcriber or DirectWhisperTranscriber(sample_rate=sample_rate)
        self._window_seconds = window_seconds
        self._interval_seconds = interval_seconds
        self._sample_rate = sample_rate
        self._diarization_enabled = enable_diarization
        if enable_diarization and config.DIARIZATION_BACKEND != "speechbrain-ecapa":
            raise ValueError(f"Unsupported DIARIZATION_BACKEND={config.DIARIZATION_BACKEND!r}")
        self._diarizer = diarizer or (OnlineSpeakerDiarizer() if enable_diarization else None)
        self._diarization_failed = False
        self._diarization_error: str | None = None
        self._frames: deque[TimestampedFrame] = deque()
        self._max_samples = max(1, int(window_seconds * sample_rate))
        self._sample_count = 0
        self._last_transcribe = 0.0
        self._last_text = ""
        self._emitted_segments: deque[tuple[str, float, float]] = deque(maxlen=200)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.latest_result: TranscriptionResult | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._emitted_segments.clear()
        self._last_text = ""
        self._diarization_failed = False
        self._diarization_error = None
        if self._diarizer is not None:
            self._diarizer.reset()
        self._thread = threading.Thread(target=self._run, name="transcription-worker", daemon=True)
        self._thread.start()
        logger.info("Transcription worker started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=config.TRANSCRIPTION_STOP_TIMEOUT_SECONDS)
        if self._thread and self._thread.is_alive():
            logger.warning(
                "Transcription worker did not stop within %.1fs; final buffered audio may be delayed",
                config.TRANSCRIPTION_STOP_TIMEOUT_SECONDS,
            )
        else:
            self._drain_audio_queue()
            if self._sample_count:
                self._transcribe_buffer()
        logger.info("Transcription worker stopped.")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._drain_audio_queue()
            now = time.monotonic()
            if self._sample_count and now - self._last_transcribe >= self._interval_seconds:
                self._last_transcribe = now
                self._transcribe_buffer()
            time.sleep(0.01)

    def _drain_audio_queue(self) -> None:
        received = False
        while True:
            try:
                frame = self._audio_queue.get_nowait()
            except queue.Empty:
                break
            self._frames.append(frame)
            self._sample_count += int(frame.data.size)
            received = True
        while self._sample_count > self._max_samples and self._frames:
            old = self._frames.popleft()
            self._sample_count -= int(old.data.size)
        if received:
            logger.info("Received audio buffer. Buffer duration: %.3f seconds", self._sample_count / self._sample_rate)

    def _transcribe_buffer(self) -> None:
        if not self._frames:
            return
        audio = np.concatenate([frame.data for frame in self._frames]).astype(np.float32, copy=False)
        buffer_duration = float(audio.size / self._sample_rate)
        audio_start_ts = self._frames[0].timestamp
        frame_duration = float(self._frames[-1].data.size / self._sample_rate)
        audio_end_ts = self._frames[-1].timestamp + frame_duration
        logger.info("Transcribing... Buffer duration: %.3f seconds", buffer_duration)
        start = time.monotonic()
        try:
            text, segments, confidence = self._transcriber.transcribe(audio)
        except Exception:  # noqa: BLE001
            logger.exception("Transcription failed")
            return
        inference_ms = (time.monotonic() - start) * 1000.0
        transcript_ts = audio_end_ts
        segment_ends = [segment.end for segment in segments if segment.end is not None]
        if segment_ends:
            transcript_ts = min(audio_start_ts + max(segment_ends), audio_end_ts)
        result = TranscriptionResult(text, transcript_ts, segments, confidence, inference_ms, buffer_duration)
        self.latest_result = result
        logger.info('Transcript:\n"%s"', text)
        logger.info("Inference time: %.2f ms", inference_ms)
        if text:
            self._put_latest(self._partial_queue, text)
            emitted = False
            for segment in segments:
                clean_text = segment.text.strip()
                if not clean_text:
                    continue
                relative_start = max(0.0, float(segment.start or 0.0))
                relative_end = max(relative_start, float(segment.end or buffer_duration))
                start_ts = min(audio_start_ts + relative_start, audio_end_ts)
                end_ts = min(audio_start_ts + relative_end, audio_end_ts)
                key = (clean_text, round(start_ts, 1), round(end_ts, 1))
                if key in self._emitted_segments:
                    continue
                start_sample = min(audio.size, max(0, int(relative_start * self._sample_rate)))
                end_sample = min(audio.size, max(start_sample, int(relative_end * self._sample_rate)))
                speaker_id, speaker_label = self._identify_speaker(audio[start_sample:end_sample])
                trace = LatencyTrace(
                    microphone_ts=self._frames[0].capture_monotonic or None,
                    queue_ts=self._frames[0].queued_monotonic or None,
                    transcript_ts=time.monotonic(),
                )
                committed = CommittedLine(
                    text=clean_text,
                    timestamp=end_ts,
                    latency_trace=trace,
                    speaker_id=speaker_id,
                    speaker_label=speaker_label,
                    start_time=start_ts,
                    end_time=end_ts,
                    transcript_confidence=confidence,
                )
                logger.info(
                    "SPEAKER speaker=%s text=%r start=%.3f end=%.3f",
                    speaker_id,
                    clean_text,
                    start_ts,
                    end_ts,
                )
                self._put_latest(self._committed_queue, committed)
                self._emitted_segments.append(key)
                emitted = True

            # Some injected/alternative transcribers may return text without
            # segment timing. Preserve the pre-diarization behaviour for them.
            if not segments and text != self._last_text:
                committed = CommittedLine(
                    text=text,
                    timestamp=result.timestamp,
                    transcript_confidence=confidence,
                )
                self._put_latest(self._committed_queue, committed)
                emitted = True
            if emitted:
                self._last_text = text

    def _identify_speaker(self, audio: np.ndarray) -> tuple[int | None, str | None]:
        if self._diarizer is None or self._diarization_failed:
            return None, None
        try:
            return self._diarizer.identify(audio, self._sample_rate)
        except Exception as exc:  # noqa: BLE001
            self._diarization_failed = True
            self._diarization_error = f"{type(exc).__name__}: {exc}"
            logger.exception(
                "Speaker diarization disabled for this session after initialization/inference failure"
            )
            return None, None

    @property
    def speakers_seen(self) -> int:
        return self._diarizer.speakers_seen if self._diarizer is not None else 0

    @property
    def diarization_active(self) -> bool:
        return self._diarizer is not None and not self._diarization_failed

    @property
    def diarization_error(self) -> str | None:
        return self._diarization_error

    @staticmethod
    def _put_latest(target: queue.Queue[Any], item: Any) -> None:
        try:
            target.put_nowait(item)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            target.put_nowait(item)

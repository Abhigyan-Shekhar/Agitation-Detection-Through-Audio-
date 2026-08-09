"""Continuous acoustic feature extraction with rolling ring buffer.

Responsibilities
----------------
* Consume ``TimestampedFrame`` objects from ``AudioPipeline.acoustic_queue``.
* Maintain a **60-second rolling ring buffer** of timestamped float32 audio.
* Every ``ACOUSTIC_HOP_SEC`` (500 ms), extract features over the last
  ``ACOUSTIC_WINDOW_SEC`` (2 s) of buffered audio.
* Apply a Silero VAD **speech mask** when computing voiced-frame-based
  features (voice ratio, pause ratio). VAD is NOT used as a gate —
  silent frames are still stored and their energy is still measured.
* Store each ``AcousticFeatureWindow`` in a time-indexed deque.
* Expose ``aggregate(start_time, end_time)`` for the fusion pipeline to
  retrieve averaged features over the time span of a completed utterance.

Design notes
------------
* Silero VAD runs frame-by-frame as a lightweight speech mask.
* Pitch estimation uses ``librosa.pyin`` — it is applied only to the
  voiced frames identified by VAD to avoid estimating pitch over silence.
* All features are sanitised to finite floats before storage.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Deque

import librosa
import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

import config
from acoustic_vocalization_detector import detect_acoustic_vocalization
from audio_pipeline import TimestampedFrame
from event_models import AcousticFeatureWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: safe float sanitiser
# ---------------------------------------------------------------------------

def _safe(value: float, default: float = 0.0) -> float:
    """Return a finite float, or ``default`` if invalid."""
    try:
        f = float(value)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Silero VAD wrapper
# ---------------------------------------------------------------------------

class SileroVAD:
    """Thin wrapper around the Silero VAD model for per-frame speech detection."""

    def __init__(self, sample_rate: int = config.SAMPLE_RATE, threshold: float = config.VAD_THRESHOLD) -> None:
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._model = None

        if torch is None:
            logger.warning("PyTorch not available — VAD mask disabled; voiced_ratio will default to 1.0")
            return

        try:
            logger.info("Loading Silero VAD model…")
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            model.reset_states()
            self._model = model
            logger.info("Silero VAD loaded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load Silero VAD: %s — voiced_ratio defaults to 1.0", exc)

    def is_speech(self, frame: np.ndarray) -> bool:
        """Return True if the frame contains speech."""
        if self._model is None:
            # Fallback: energy-based heuristic
            return bool(np.mean(np.abs(frame)) > 1e-4)

        tensor = torch.from_numpy(frame.astype(np.float32))
        with torch.no_grad():
            prob = self._model(tensor, self._sample_rate)
        return float(prob) >= self._threshold


# ---------------------------------------------------------------------------
# Rolling audio ring buffer
# ---------------------------------------------------------------------------

@dataclass
class _AudioRecord:
    data: np.ndarray   # float32
    timestamp: float   # Unix timestamp of the frame leading edge
    is_speech: bool    # Silero VAD label


class AudioRingBuffer:
    """Fixed-duration rolling buffer of timestamped audio frames.

    Parameters
    ----------
    max_seconds:
        Maximum history retained. Older frames are evicted automatically.
    frame_size:
        Samples per frame (must match the pipeline's frame_size).
    sample_rate:
        Audio sample rate in Hz.
    """

    def __init__(
        self,
        max_seconds: float = config.AUDIO_RING_BUFFER_SEC,
        frame_size: int = config.FRAME_SIZE,
        sample_rate: int = config.SAMPLE_RATE,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        fps = sample_rate / frame_size                        # frames per second
        max_frames = int(max_seconds * fps) + 1
        self._buffer: Deque[_AudioRecord] = collections.deque(maxlen=max_frames)

    def append(self, frame: TimestampedFrame, is_speech: bool) -> None:
        self._buffer.append(_AudioRecord(
            data=frame.data,
            timestamp=frame.timestamp,
            is_speech=is_speech,
        ))

    def slice(self, start_time: float, end_time: float) -> list[_AudioRecord]:
        """Return all records whose timestamp falls in [start_time, end_time]."""
        return [r for r in self._buffer if start_time <= r.timestamp <= end_time]

    def latest_window(self, window_sec: float) -> list[_AudioRecord]:
        """Return the most recent ``window_sec`` seconds of audio."""
        cutoff = time.time() - window_sec
        return [r for r in self._buffer if r.timestamp >= cutoff]

    def oldest_timestamp(self) -> float | None:
        return self._buffer[0].timestamp if self._buffer else None

    def newest_timestamp(self) -> float | None:
        return self._buffer[-1].timestamp if self._buffer else None


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------

class AcousticExtractor:
    """Extracts ``AcousticFeatureWindow`` from a list of audio records."""

    def __init__(self, sample_rate: int = config.SAMPLE_RATE) -> None:
        self._sr = sample_rate

    def extract(self, records: list[_AudioRecord], window_start: float, window_end: float) -> AcousticFeatureWindow:
        if not records:
            return AcousticFeatureWindow(start_time=window_start, end_time=window_end)

        audio = np.concatenate([r.data for r in records]).astype(np.float32)

        # --- Energy ---
        rms_frames = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        rms_mean = _safe(float(np.mean(rms_frames)))
        rms_max = _safe(float(np.max(rms_frames)))
        if rms_frames.size > 1:
            x = np.arange(rms_frames.size, dtype=np.float32)
            rms_slope = _safe(float(np.polyfit(x, rms_frames, 1)[0]))
        else:
            rms_slope = 0.0

        # --- Zero crossing rate ---
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=512, hop_length=256)[0]
        zcr_mean = _safe(float(np.mean(zcr)))

        # --- Spectral features ---
        spectral_centroid = _safe(float(np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=self._sr, n_fft=512, hop_length=256)
        )))
        spectral_rolloff = _safe(float(np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=self._sr, n_fft=512, hop_length=256)
        )))

        # --- HNR (approximation via harmonics-to-noise estimate) ---
        try:
            harmonic = librosa.effects.harmonic(audio)
            noise = audio - harmonic
            h_rms = float(np.sqrt(np.mean(harmonic ** 2)) + 1e-10)
            n_rms = float(np.sqrt(np.mean(noise ** 2)) + 1e-10)
            hnr = _safe(float(20 * np.log10(h_rms / n_rms)))
        except Exception:  # noqa: BLE001
            hnr = 0.0

        vocalization = detect_acoustic_vocalization(audio, self._sr)

        # --- Voice activity mask ---
        voiced_records = [r for r in records if r.is_speech]
        voiced_ratio = _safe(len(voiced_records) / max(len(records), 1))
        pause_ratio = _safe(1.0 - voiced_ratio)

        # --- Clipping ratio ---
        clipping_ratio = _safe(float(np.mean(np.abs(audio) >= 0.99)))

        # --- Pitch (voiced frames only) ---
        if voiced_records:
            voiced_audio = np.concatenate([r.data for r in voiced_records]).astype(np.float32)
            try:
                f0, voiced_flag, _ = librosa.pyin(
                    voiced_audio,
                    fmin=librosa.note_to_hz("C2"),
                    fmax=librosa.note_to_hz("C7"),
                    sr=self._sr,
                )
                f0_voiced = f0[voiced_flag & np.isfinite(f0)]
                if f0_voiced.size >= 2:
                    pitch_median = _safe(float(np.median(f0_voiced)))
                    pitch_range = _safe(float(np.ptp(f0_voiced)))   # max - min
                    pitch_variance = _safe(float(np.var(f0_voiced)))
                else:
                    pitch_median = pitch_range = pitch_variance = 0.0
            except Exception:  # noqa: BLE001
                pitch_median = pitch_range = pitch_variance = 0.0
        else:
            pitch_median = pitch_range = pitch_variance = 0.0

        return AcousticFeatureWindow(
            start_time=window_start,
            end_time=window_end,
            rms_mean=rms_mean,
            rms_max=rms_max,
            rms_slope=rms_slope,
            pitch_median=pitch_median,
            pitch_range=pitch_range,
            pitch_variance=pitch_variance,
            zcr_mean=zcr_mean,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            harmonic_to_noise_ratio=hnr,
            non_speech_vocalization_score=vocalization.score,
            non_speech_vocalization_label=vocalization.label,
            non_speech_vocalization_evidence=vocalization.evidence or None,
            voiced_ratio=voiced_ratio,
            pause_ratio=pause_ratio,
            clipping_ratio=clipping_ratio,
        )


# ---------------------------------------------------------------------------
# Acoustic worker
# ---------------------------------------------------------------------------

class AcousticWorker:
    """Background thread that continuously extracts feature windows.

    Stores windows in a time-indexed deque that the fusion pipeline can
    query via ``aggregate()``.

    Parameters
    ----------
    acoustic_queue:
        Source of ``TimestampedFrame`` objects from ``AudioPipeline``.
    window_sec:
        Length of each feature extraction window.
    hop_sec:
        How frequently to extract a new window.
    ring_buffer_sec:
        How much audio history to retain in memory.
    """

    def __init__(
        self,
        acoustic_queue: queue.Queue[TimestampedFrame],
        window_sec: float = config.ACOUSTIC_WINDOW_SEC,
        hop_sec: float = config.ACOUSTIC_HOP_SEC,
        ring_buffer_sec: float = config.AUDIO_RING_BUFFER_SEC,
        on_window: Callable[[AcousticFeatureWindow], None] | None = None,
    ) -> None:
        self._queue = acoustic_queue
        self._window_sec = window_sec
        self._hop_sec = hop_sec

        self._vad = SileroVAD()
        self._ring = AudioRingBuffer(max_seconds=ring_buffer_sec)
        self._extractor = AcousticExtractor()
        self._on_window = on_window

        # Time-indexed window store (deque so old windows auto-expire)
        max_windows = int(ring_buffer_sec / hop_sec) + 2
        self._windows: Deque[AcousticFeatureWindow] = collections.deque(maxlen=max_windows)
        self._lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_extraction_time: float = 0.0
        self._windows_extracted: int = 0
        self._last_extraction_ms: float = 0.0
        self._total_extraction_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="acoustic-worker", daemon=True
        )
        self._thread.start()
        logger.info("AcousticWorker started")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info(
            "AcousticWorker stopped — windows extracted: %d", self._windows_extracted
        )

    def aggregate(self, start_time: float, end_time: float) -> AcousticFeatureWindow | None:
        """Return the mean of all windows that overlap [start_time, end_time].

        Uses precomputed feature windows so dashboard inference never blocks
        on synchronous librosa feature extraction.
        """
        with self._lock:
            relevant = [
                w for w in self._windows
                if w.start_time <= end_time and w.end_time >= start_time
            ]
        if not relevant:
            return None
        feat = self._average_windows(relevant, start_time, end_time)
        logger.debug(
            "Acoustic aggregate used %d precomputed windows over %.2fs",
            len(relevant),
            end_time - start_time,
        )
        return feat

    def latest_window(self) -> AcousticFeatureWindow | None:
        with self._lock:
            return self._windows[-1] if self._windows else None

    @property
    def windows_extracted(self) -> int:
        return self._windows_extracted

    @property
    def last_extraction_ms(self) -> float:
        return self._last_extraction_ms

    @property
    def average_extraction_ms(self) -> float:
        if self._windows_extracted <= 0:
            return 0.0
        return self._total_extraction_ms / self._windows_extracted

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # Drain the incoming queue into the ring buffer
            self._drain_queue()

            # Trigger extraction every hop_sec
            now = time.time()
            if now - self._last_extraction_time >= self._hop_sec:
                records = self._ring.latest_window(self._window_sec)
                if records:
                    window_end = now
                    window_start = window_end - self._window_sec
                    extract_start = time.monotonic()
                    feat = self._extractor.extract(records, window_start, window_end)
                    extract_ms = (time.monotonic() - extract_start) * 1000.0
                    logger.debug("Acoustic feature extraction window took %.2f ms", extract_ms)
                    self._store_window(feat, extract_ms)

            time.sleep(0.010)   # ~10 ms yield

    def _store_window(self, feat: AcousticFeatureWindow, extract_ms: float) -> None:
        with self._lock:
            self._windows.append(feat)
        self._last_extraction_time = time.time()
        self._windows_extracted += 1
        self._last_extraction_ms = extract_ms
        self._total_extraction_ms += extract_ms

        if self._on_window is not None:
            try:
                self._on_window(feat)
            except Exception:  # noqa: BLE001
                logger.exception("Acoustic window callback failed")

    def _drain_queue(self) -> None:
        while True:
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                return
            is_speech = self._vad.is_speech(frame.data)
            self._ring.append(frame, is_speech)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _average_windows(
        windows: list[AcousticFeatureWindow],
        start_time: float,
        end_time: float,
    ) -> AcousticFeatureWindow:
        """Compute element-wise mean across a list of feature windows."""
        text_fields = {"non_speech_vocalization_label", "non_speech_vocalization_evidence"}
        fields = [
            f for f in AcousticFeatureWindow.__dataclass_fields__
            if f not in ("start_time", "end_time") and f not in text_fields
        ]
        averaged = {f: float(np.mean([getattr(w, f) for w in windows])) for f in fields}
        strongest = max(windows, key=lambda w: w.non_speech_vocalization_score)
        averaged["non_speech_vocalization_label"] = strongest.non_speech_vocalization_label
        averaged["non_speech_vocalization_evidence"] = strongest.non_speech_vocalization_evidence
        return AcousticFeatureWindow(start_time=start_time, end_time=end_time, **averaged)

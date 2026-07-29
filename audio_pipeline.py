"""Audio capture and fan-out to downstream processing queues.

Responsibilities
----------------
* Open a sounddevice InputStream at 16 kHz mono float32.
* On every callback, put a copy of the raw float32 frame on two queues:
    - ``acoustic_queue``  → acoustic feature worker
    - ``wlk_queue``       → WhisperLiveKit PCM16 sender
* Track frame timestamps so downstream workers can align features with
  transcript segments.
* Expose start() / stop() for the dashboard to call.

What this module does NOT do
-----------------------------
* No VAD gating — WhisperLiveKit handles its own VAD internally.
* No overlapping window buffering — that moved to acoustic_features.py.
* No transcription — that moved to whisperlivekit_client.py.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import NamedTuple

import numpy as np

try:
    import sounddevice as sd  # type: ignore[import-untyped]
except ImportError:
    sd = None  # type: ignore[assignment]

import config

logger = logging.getLogger(__name__)

_SILENCE_RMS_THRESHOLD = 1e-5


class TimestampedFrame(NamedTuple):
    """A single audio callback chunk with wall-clock metadata."""

    data: np.ndarray   # float32, shape (frame_size,)
    timestamp: float   # Unix timestamp of the frame's leading edge
    capture_monotonic: float = 0.0
    queued_monotonic: float = 0.0
    frame_index: int = 0


class AudioPipeline:
    """Captures microphone audio and fans frames out to downstream workers.

    Parameters
    ----------
    sample_rate:
        Capture sample rate in Hz. Must match WhisperLiveKit's expected input.
    frame_size:
        Number of samples per sounddevice callback (one ``TimestampedFrame``).
    max_queue_size:
        Maximum frames held in each output queue before old frames are dropped.
        At 512 samples / 16 000 Hz = ~32 ms per frame, 1 second = ~31 frames.
        Default 200 gives roughly a 6-second slack buffer.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        frame_size: int = config.FRAME_SIZE,
        max_queue_size: int = 200,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        # Output queues — bounded to prevent unbounded memory growth
        self.acoustic_queue: queue.Queue[TimestampedFrame] = queue.Queue(
            maxsize=max_queue_size
        )
        self.wlk_queue: queue.Queue[TimestampedFrame] = queue.Queue(
            maxsize=max_queue_size
        )

        self._stream: sd.InputStream | None = None
        self._is_running: bool = False
        self._dropped_frames: int = 0

        # Monotonic frame counter for diagnostics
        self._frame_index: int = 0
        self._last_callback_stats: dict[str, float | int | str | bool] = {}
        self._input_device: str | int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def start(self) -> None:
        """Open the microphone stream and begin fanning frames to queues."""
        if self._is_running:
            logger.warning("AudioPipeline.start() called while already running")
            return

        if sd is None:
            raise RuntimeError(
                "sounddevice is not installed. Run: pip install sounddevice"
            )

        self._dropped_frames = 0
        self._frame_index = 0
        self._flush_queues()
        self._is_running = True

        self._input_device = self._resolve_input_device()
        self._log_input_device_config()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=config.CHANNELS,
            blocksize=self.frame_size,
            dtype=config.DTYPE,
            device=self._input_device,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._log_stream_config()
        logger.info(
            "AudioPipeline started — sample_rate=%d frame_size=%d",
            self.sample_rate,
            self.frame_size,
        )

    def stop(self) -> None:
        """Stop microphone capture."""
        if not self._is_running:
            return

        self._is_running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        logger.info(
            "AudioPipeline stopped — total dropped frames: %d acoustic_q=%d wlk_q=%d",
            self._dropped_frames,
            self.acoustic_queue.qsize(),
            self.wlk_queue.qsize(),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """sounddevice calls this on every audio block (runs in a C thread)."""
        if status:
            logger.debug("sounddevice status: %s", status)

        self._log_callback_stats(indata, frames)

        audio = indata[:, 0].copy()  # mono, float32
        ts = time.time()
        capture_monotonic = time.monotonic()
        self._frame_index += 1
        frame = TimestampedFrame(
            data=audio,
            timestamp=ts,
            capture_monotonic=capture_monotonic,
            queued_monotonic=time.monotonic(),
            frame_index=self._frame_index,
        )

        # Fan out to both queues — drop if full rather than block the callback
        for q in (self.acoustic_queue, self.wlk_queue):
            try:
                q.put_nowait(frame)
            except queue.Full:
                self._dropped_frames += 1
                logger.debug(
                    "Queue full — frame %d dropped (total dropped: %d)",
                    self._frame_index,
                    self._dropped_frames,
                )

    def _capture_callback_audio(self, indata: np.ndarray) -> None:
        """Compatibility no-op for removed callback WAV diagnostics."""
        return None

    def _log_audio_stage(self, *args: object, **kwargs: object) -> None:
        """Compatibility no-op for removed per-stage audio diagnostics."""
        return None

    def _resolve_input_device(self) -> str | int | None:
        """Return the concrete input device passed to sounddevice.InputStream."""
        assert sd is not None
        if config.AUDIO_INPUT_DEVICE is not None:
            return self._coerce_input_device(config.AUDIO_INPUT_DEVICE)

        default_device = sd.default.device
        try:
            return default_device[0]
        except (TypeError, IndexError, KeyError):
            return default_device

    def _coerce_input_device(self, device: str | int) -> str | int:
        """Convert numeric environment values to sounddevice input-device indexes."""
        if isinstance(device, str):
            value = device.strip()
            if "," in value:
                value = value.split(",", 1)[0].strip()
            if value.isdigit():
                return int(value)
        return device

    def _log_input_device_config(self) -> None:
        """Log the exact input device that will be passed to InputStream."""
        assert sd is not None
        try:
            devices = sd.query_devices()
            default_device = sd.default.device
            try:
                default_input = default_device[0]
            except (TypeError, IndexError, KeyError):
                default_input = default_device
            selected = sd.query_devices(self._input_device, "input")
            logger.info(
                "Audio input device selected — requested_config=%r default_device=%r default_input=%r resolved_input=%r name=%s hostapi=%s max_input_channels=%s default_samplerate=%s",
                config.AUDIO_INPUT_DEVICE,
                default_device,
                default_input,
                self._input_device,
                selected.get("name"),
                selected.get("hostapi"),
                selected.get("max_input_channels"),
                selected.get("default_samplerate"),
            )
            logger.debug("Available audio devices: %s", devices)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not query sounddevice input devices: %s", exc)

    def _log_stream_config(self) -> None:
        """Log the actual PortAudio stream configuration after start."""
        if self._stream is None:
            return
        logger.info(
            "Audio InputStream started — requested_config=%r resolved_device=%r samplerate=%s channels=%d dtype=%s blocksize=%d latency=%s",
            config.AUDIO_INPUT_DEVICE,
            self._input_device,
            getattr(self._stream, "samplerate", self.sample_rate),
            config.CHANNELS,
            config.DTYPE,
            self.frame_size,
            getattr(self._stream, "latency", "unknown"),
        )

    def _log_callback_stats(self, indata: np.ndarray, frames: int) -> None:
        """Log raw callback-level audio statistics before any processing."""
        finite = bool(np.all(np.isfinite(indata)))
        sample_min = float(np.nanmin(indata)) if indata.size else 0.0
        sample_max = float(np.nanmax(indata)) if indata.size else 0.0
        rms = float(np.sqrt(np.nanmean(indata ** 2))) if indata.size else 0.0
        peak = float(np.nanmax(np.abs(indata))) if indata.size else 0.0
        clipped = int(np.sum(np.abs(indata) >= 0.99)) if indata.size else 0
        self._last_callback_stats = {
            "frames": frames,
            "samples": int(indata.size),
            "dtype": str(indata.dtype),
            "rms": rms,
            "peak": peak,
            "min": sample_min,
            "max": sample_max,
            "finite": finite,
            "clipped": clipped,
        }
        logger.info(
            "Audio callback raw stats — frames=%d samples=%d dtype=%s rms=%.8f peak=%.8f min=%.8f max=%.8f finite=%s clipped=%d",
            frames,
            int(indata.size),
            indata.dtype,
            rms,
            peak,
            sample_min,
            sample_max,
            finite,
            clipped,
        )
        if not finite:
            logger.warning("Audio callback received non-finite samples")
        elif peak == 0.0:
            logger.warning("Audio callback received all-zero samples")
        elif rms < _SILENCE_RMS_THRESHOLD:
            logger.warning("Audio callback amplitude is extremely low: rms=%.8f peak=%.8f", rms, peak)

    def _flush_queues(self) -> None:
        """Drain all queues before (re)starting to avoid stale data."""
        for q in (self.acoustic_queue, self.wlk_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break


# ---------------------------------------------------------------------------
# Quick smoke-test entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time

    logging.basicConfig(level=logging.INFO)
    pipeline = AudioPipeline()
    pipeline.start()
    print("Listening for 5 s — check that frames appear on both queues…")
    try:
        _time.sleep(5)
    finally:
        pipeline.stop()

    acoustic_size = pipeline.acoustic_queue.qsize()
    wlk_size = pipeline.wlk_queue.qsize()
    print(
        f"acoustic_queue={acoustic_size} frames, "
        f"wlk_queue={wlk_size} frames, "
        f"dropped={pipeline.dropped_frames}"
    )

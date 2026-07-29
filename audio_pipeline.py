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

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=config.CHANNELS,
            blocksize=self.frame_size,
            dtype=config.DTYPE,
            callback=self._audio_callback,
        )
        self._stream.start()
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

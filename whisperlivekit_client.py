"""WhisperLiveKit WebSocket client.

Responsibilities
----------------
* Consume ``TimestampedFrame`` objects from ``AudioPipeline.wlk_queue``.
* Convert float32 PCM to signed 16-bit little-endian bytes.
* Stream PCM bytes to the local WhisperLiveKit WebSocket server.
* Parse JSON messages from WLK and classify them as:
    - ``partial``   → live caption (put on ``partial_queue``)
    - ``committed`` → stable transcript lines (put on ``committed_queue``)
* Reconnect a bounded number of times after a connection drop.
* On ``stop()``, flush any remaining audio before closing.

WhisperLiveKit message protocol (--pcm-input mode)
---------------------------------------------------
Inbound (to server):  raw PCM16 bytes in chunks (no framing required)
Outbound (from server): JSON objects, e.g.:
    {"type": "partial",   "text": "Why can't I go"}
    {"type": "committed", "text": "Why can't I go home?"}

Note: WLK 0.x may use slightly different field names. The parser below
normalises both ``type``/``status`` field variants.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Any

import numpy as np

try:
    import websockets  # type: ignore[import-untyped]
    from websockets.exceptions import ConnectionClosed
except ImportError:
    websockets = None  # type: ignore[assignment]
    ConnectionClosed = Exception  # type: ignore[assignment,misc]

import config
from audio_pipeline import TimestampedFrame
from event_models import CommittedLine

logger = logging.getLogger(__name__)

# How many bytes to send per WebSocket message (≈ 40 ms of audio)
_SEND_CHUNK_BYTES: int = int(config.SAMPLE_RATE * 0.040 * 2)  # 16-bit = 2 bytes/sample
_RECONNECT_DELAY_SEC: float = 2.0
_MAX_RECONNECT_ATTEMPTS: int = 5


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float32 audio array to signed 16-bit little-endian PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


class WhisperLiveKitClient:
    """Streams PCM audio to WLK and distributes transcript events to queues.

    Parameters
    ----------
    wlk_queue:
        Source of ``TimestampedFrame`` frames from ``AudioPipeline``.
    partial_queue:
        Destination for real-time partial caption strings (str).
    committed_queue:
        Destination for finalised ``CommittedLine`` objects.
    url:
        WebSocket URL of the running WLK server.
    """

    def __init__(
        self,
        wlk_queue: queue.Queue[TimestampedFrame],
        partial_queue: queue.Queue[str],
        committed_queue: queue.Queue[CommittedLine],
        url: str = config.WLK_URL,
    ) -> None:
        if websockets is None:
            raise ImportError(
                "websockets is not installed. Run: pip install websockets"
            )

        self._wlk_queue = wlk_queue
        self._partial_queue = partial_queue
        self._committed_queue = committed_queue
        self._url = url

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: Any | None = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._last_error: Exception | None = None

        self._bytes_sent: int = 0
        self._partial_count: int = 0
        self._committed_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the asyncio event loop in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("WhisperLiveKitClient already running")
            return

        self._stop_event.clear()
        self._connected_event.clear()
        self._last_error = None
        self._thread = threading.Thread(
            target=self._run_event_loop, name="wlk-client", daemon=True
        )
        self._thread.start()
        logger.info("WhisperLiveKitClient thread started → %s", self._url)

    def wait_until_connected(self, timeout: float = 10.0) -> None:
        """Block until the websocket connects or raise a startup error."""
        if not self._connected_event.wait(timeout=timeout):
            if self._last_error is not None:
                raise RuntimeError(
                    f"WLK websocket failed to connect: {self._last_error}"
                )
            raise TimeoutError(
                f"WLK websocket did not connect to {self._url} "
                f"within {timeout:.1f}s"
            )

    def stop(self) -> None:
        """Signal the client to flush remaining audio and shut down."""
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            if self._ws is not None:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread:
            self._thread.join(timeout=5.0)
        self._connected_event.clear()
        logger.info(
            "WhisperLiveKitClient stopped — bytes_sent=%d partial=%d committed=%d",
            self._bytes_sent,
            self._partial_count,
            self._committed_count,
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "bytes_sent": self._bytes_sent,
            "partial_count": self._partial_count,
            "committed_count": self._committed_count,
        }

    # ------------------------------------------------------------------
    # Asyncio internals
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._connect_loop())
        finally:
            loop.close()

    async def _connect_loop(self) -> None:
        """Connect to WLK and retry a bounded number of times while running."""
        attempts = 0
        while not self._stop_event.is_set():
            try:
                logger.info("Connecting to WLK at %s", self._url)
                async with websockets.connect(self._url) as ws:
                    self._ws = ws
                    attempts = 0
                    self._last_error = None
                    self._connected_event.set()
                    logger.info("WLK connection established")
                    try:
                        await asyncio.gather(
                            self._send_loop(ws),
                            self._recv_loop(ws),
                        )
                    finally:
                        self._ws = None
            except ConnectionClosed as exc:
                self._connected_event.clear()
                if self._stop_event.is_set():
                    break
                attempts += 1
                self._last_error = exc
                if attempts > _MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "WLK connection closed after %d retries: %s",
                        attempts - 1,
                        exc,
                    )
                    break
                logger.warning(
                    "WLK connection closed: %s — reconnecting in %.1fs (%d/%d)",
                    exc,
                    _RECONNECT_DELAY_SEC,
                    attempts,
                    _MAX_RECONNECT_ATTEMPTS,
                )
                await asyncio.sleep(_RECONNECT_DELAY_SEC)
            except OSError as exc:
                self._connected_event.clear()
                if self._stop_event.is_set():
                    break
                attempts += 1
                self._last_error = exc
                if attempts > _MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "WLK connection failed after %d retries: %s",
                        attempts - 1,
                        exc,
                    )
                    break
                logger.warning(
                    "WLK connection failed: %s — retrying in %.1fs (%d/%d)",
                    exc,
                    _RECONNECT_DELAY_SEC,
                    attempts,
                    _MAX_RECONNECT_ATTEMPTS,
                )
                await asyncio.sleep(_RECONNECT_DELAY_SEC)
            except Exception as exc:  # noqa: BLE001
                self._connected_event.clear()
                if self._stop_event.is_set():
                    break
                attempts += 1
                self._last_error = exc
                if attempts > _MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "WLK websocket error after %d retries: %s",
                        attempts - 1,
                        exc,
                    )
                    break
                logger.warning(
                    "WLK websocket error: %s — retrying in %.1fs (%d/%d)",
                    exc,
                    _RECONNECT_DELAY_SEC,
                    attempts,
                    _MAX_RECONNECT_ATTEMPTS,
                )
                await asyncio.sleep(_RECONNECT_DELAY_SEC)
        self._connected_event.clear()

    async def _send_loop(self, ws: Any) -> None:
        """Drain wlk_queue and stream PCM16 bytes to WLK."""
        pcm_buffer = bytearray()

        while not self._stop_event.is_set():
            # Drain all available frames without blocking
            try:
                while True:
                    frame: TimestampedFrame = self._wlk_queue.get_nowait()
                    pcm_buffer.extend(float32_to_pcm16(frame.data))
            except queue.Empty:
                pass

            if len(pcm_buffer) >= _SEND_CHUNK_BYTES:
                chunk = bytes(pcm_buffer[:_SEND_CHUNK_BYTES])
                del pcm_buffer[:_SEND_CHUNK_BYTES]
                try:
                    await ws.send(chunk)
                    self._bytes_sent += len(chunk)
                except ConnectionClosed:
                    raise

            await asyncio.sleep(0.010)  # yield — ~10 ms polling interval

        # Flush remainder on shutdown
        if pcm_buffer:
            try:
                await ws.send(bytes(pcm_buffer))
                self._bytes_sent += len(pcm_buffer)
            except ConnectionClosed:
                pass

    async def _recv_loop(self, ws: Any) -> None:
        """Receive JSON messages from WLK and route to the correct queue."""
        async for raw in ws:
            if self._stop_event.is_set():
                break
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Non-JSON WLK message: %r", raw)
                continue

            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route a parsed WLK message to the appropriate output queue."""
        # WLK may use "type" or "status" depending on version
        msg_type = msg.get("type") or msg.get("status") or ""
        text: str = msg.get("text", "") or msg.get("transcript", "") or ""

        if not text:
            return

        if msg_type in ("partial", "interim", "processing"):
            try:
                self._partial_queue.put_nowait(text)
                self._partial_count += 1
            except queue.Full:
                pass  # partial captions are display-only; drop if dashboard is slow

        elif msg_type in ("committed", "final", "complete", "completed"):
            line = CommittedLine(text=text.strip(), timestamp=time.time())
            try:
                self._committed_queue.put_nowait(line)
                self._committed_count += 1
                logger.debug("Committed: %r", text)
            except queue.Full:
                logger.warning("committed_queue full — committed line dropped")

        else:
            # Some WLK versions emit a bare dict with only a "text" field
            # once a segment stabilises; treat that as committed.
            if text and msg_type == "":
                line = CommittedLine(text=text.strip(), timestamp=time.time())
                try:
                    self._committed_queue.put_nowait(line)
                    self._committed_count += 1
                except queue.Full:
                    pass
            else:
                logger.debug("Unknown WLK message type %r: %r", msg_type, msg)

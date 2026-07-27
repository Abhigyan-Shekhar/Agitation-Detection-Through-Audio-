"""WhisperLiveKit WebSocket client.

The client owns one background thread, one asyncio event loop, one websocket
connection at a time, one send loop, and one receive loop.  It connects only to
``config.WLK_URL`` and retries connection failures a bounded number of times.
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

_SEND_CHUNK_BYTES: int = int(config.SAMPLE_RATE * 0.040 * 2)
_RECONNECT_DELAY_SEC: float = 2.0
_MAX_RECONNECT_ATTEMPTS: int = 5


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float32 audio array to signed 16-bit little-endian PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return pcm.tobytes()


class WhisperLiveKitClient:
    """Stream PCM audio to WLK and route transcript events to queues."""

    def __init__(
        self,
        wlk_queue: queue.Queue[TimestampedFrame],
        partial_queue: queue.Queue[str],
        committed_queue: queue.Queue[CommittedLine] | list[queue.Queue[CommittedLine]],
        url: str = config.WLK_URL,
    ) -> None:
        if websockets is None:
            raise ImportError("websockets is not installed. Run: pip install websockets")

        self._wlk_queue = wlk_queue
        self._partial_queue = partial_queue
        if isinstance(committed_queue, list):
            self._committed_queues = committed_queue
        else:
            self._committed_queues = [committed_queue]
        self._url = url

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: Any | None = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._last_error: BaseException | None = None

        self._bytes_sent = 0
        self._partial_count = 0
        self._committed_count = 0
        self._emitted_line_keys: set[tuple[Any, ...]] = set()

    def start(self) -> None:
        """Start the single websocket background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("WhisperLiveKitClient already running")
            return

        self._stop_event.clear()
        self._connected_event.clear()
        self._last_error = None
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="wlk-client",
            daemon=True,
        )
        self._thread.start()
        logger.info("WhisperLiveKitClient thread started → %s", self._url)

    def wait_until_connected(self, timeout: float = 10.0) -> None:
        """Wait until the websocket connects or raise a startup error."""
        if self._connected_event.wait(timeout=timeout):
            return
        if self._last_error is not None:
            raise RuntimeError(f"WLK websocket failed to connect: {self._last_error}")
        raise TimeoutError(
            f"WLK websocket did not connect to {self._url} within {timeout:.1f}s"
        )

    def stop(self) -> None:
        """Close the websocket and join the background thread."""
        self._stop_event.set()
        self._connected_event.clear()
        if self._loop and self._loop.is_running() and self._ws is not None:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
        if self._thread:
            self._thread.join(timeout=5.0)
        self._ws = None
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

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_with_retries())
        finally:
            self._connected_event.clear()
            self._ws = None
            loop.close()
            self._loop = None

    async def _run_with_retries(self) -> None:
        attempts = 0
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
                attempts = 0
            except Exception as exc:  # noqa: BLE001
                self._connected_event.clear()
                self._last_error = exc
                if self._stop_event.is_set():
                    break
                attempts += 1
                if attempts > _MAX_RECONNECT_ATTEMPTS:
                    logger.error(
                        "WLK websocket failed after %d reconnect attempts: %s",
                        _MAX_RECONNECT_ATTEMPTS,
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

    async def _connect_once(self) -> None:
        logger.info("Connecting to WLK at %s", self._url)
        async with websockets.connect(self._url) as ws:
            self._ws = ws
            self._last_error = None
            self._connected_event.set()
            logger.info("WLK connection established")
            try:
                send_task = asyncio.create_task(self._send_loop(ws))
                recv_task = asyncio.create_task(self._recv_loop(ws))
                done, pending = await asyncio.wait(
                    {send_task, recv_task},
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                for task in pending:
                    task.cancel()
                for task in done:
                    task.result()
            finally:
                self._connected_event.clear()
                self._ws = None

    async def _send_loop(self, ws: Any) -> None:
        """Drain queued audio and stream PCM16 chunks to WLK."""
        pcm_buffer = bytearray()
        while not self._stop_event.is_set():
            try:
                while True:
                    frame = self._wlk_queue.get_nowait()
                    pcm_buffer.extend(float32_to_pcm16(frame.data))
            except queue.Empty:
                pass

            while len(pcm_buffer) >= _SEND_CHUNK_BYTES:
                chunk = bytes(pcm_buffer[:_SEND_CHUNK_BYTES])
                del pcm_buffer[:_SEND_CHUNK_BYTES]
                await ws.send(chunk)
                self._bytes_sent += len(chunk)

            await asyncio.sleep(0.010)

        if pcm_buffer:
            try:
                await ws.send(bytes(pcm_buffer))
                self._bytes_sent += len(pcm_buffer)
            except ConnectionClosed:
                pass

    async def _recv_loop(self, ws: Any) -> None:
        """Receive WLK JSON messages and route transcript events."""
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
        msg_type = msg.get("type") or msg.get("status") or ""
        text: str = msg.get("text", "") or msg.get("transcript", "") or ""

        buffer_text = msg.get("buffer_transcription")
        if buffer_text is not None:
            self._put_partial(str(buffer_text))

        if "lines" in msg:
            self._dispatch_lines(msg.get("lines", []))
            return

        if msg_type == "diff":
            self._dispatch_lines(msg.get("new_lines", []))
            return

        if msg_type in ("partial", "interim", "processing"):
            if text:
                self._put_partial(text)
            return

        if msg_type in ("committed", "final", "complete", "completed", ""):
            if text:
                self._put_committed_text(text)
            return

        if msg_type in ("config", "ready_to_stop", "no_audio_detected"):
            return

        logger.debug("Unknown WLK message type %r: %r", msg_type, msg)

    def _dispatch_lines(self, lines: Any) -> None:
        """Publish only newly committed WLK line objects."""
        if not isinstance(lines, list):
            logger.debug("Ignoring malformed WLK lines payload: %r", lines)
            return

        for raw_line in lines:
            if not isinstance(raw_line, dict):
                continue
            if raw_line.get("speaker") == -2:
                continue
            text = str(raw_line.get("text") or "").strip()
            if not text:
                continue
            line_key = (
                raw_line.get("speaker"),
                raw_line.get("start"),
                raw_line.get("end"),
                text,
            )
            if line_key in self._emitted_line_keys:
                continue
            self._emitted_line_keys.add(line_key)
            self._put_committed_text(text)

    def _put_partial(self, text: str) -> None:
        """Publish live, replaceable buffer text to the dashboard."""
        try:
            self._partial_queue.put_nowait(text)
            self._partial_count += 1
        except queue.Full:
            pass

    def _put_committed_text(self, text: str) -> None:
        """Publish a committed transcript line to all downstream consumers."""
        line = CommittedLine(text=text.strip(), timestamp=time.time())
        delivered = False
        for committed_queue in self._committed_queues:
            try:
                committed_queue.put_nowait(line)
                delivered = True
            except queue.Full:
                logger.warning("committed_queue full — committed line dropped")

        if delivered:
            self._committed_count += 1

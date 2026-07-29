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
import wave
import threading
import time
from pathlib import Path
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
from event_models import CommittedLine, LatencyTrace

logger = logging.getLogger(__name__)

_SEND_CHUNK_BYTES: int = int(config.SAMPLE_RATE * 0.040 * 2)
_RECONNECT_DELAY_SEC: float = 2.0
_MAX_RECONNECT_ATTEMPTS: int = 5
_DEBUG_CAPTURE_SECONDS: float = 5.0
_DEBUG_CAPTURE_PATH = Path("debug_capture.wav")


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
        committed_queue: queue.Queue[CommittedLine],
        url: str = config.WLK_URL,
    ) -> None:
        if websockets is None:
            raise ImportError("websockets is not installed. Run: pip install websockets")

        self._wlk_queue = wlk_queue
        self._partial_queue = partial_queue
        self._committed_queue = committed_queue
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
        self._last_sent_trace: LatencyTrace | None = None
        self._send_latency_ms_total = 0.0
        self._send_latency_samples = 0
        self._messages_received = 0
        self._last_committed_keys: set[tuple[str, str, str, str]] = set()
        self._config_received_event = threading.Event()
        self._received_config = False
        self._debug_capture_samples: list[np.ndarray] = []
        self._debug_capture_written = False

    def start(self) -> None:
        """Start the single websocket background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("WhisperLiveKitClient already running")
            return

        self._stop_event.clear()
        self._connected_event.clear()
        self._config_received_event.clear()
        self._received_config = False
        self._last_error = None
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="wlk-client",
            daemon=True,
        )
        self._thread.start()
        logger.info("WhisperLiveKitClient thread started → %s", self._url)

    def wait_until_connected(self, timeout: float = 10.0) -> None:
        """Wait until the websocket config arrives or raise a startup error."""
        if not self._connected_event.wait(timeout=timeout):
            if self._last_error is not None:
                raise RuntimeError(f"WLK websocket failed to connect: {self._last_error}")
            raise TimeoutError(
                f"WLK websocket did not connect to {self._url} within {timeout:.1f}s"
            )
        remaining = max(0.1, timeout)
        if self._config_received_event.wait(timeout=remaining):
            return
        raise TimeoutError(
            f"WLK websocket connected but did not send config within {timeout:.1f}s"
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
            "WhisperLiveKitClient stopped — bytes_sent=%d messages=%d partial=%d committed=%d",
            self._bytes_sent,
            self._messages_received,
            self._partial_count,
            self._committed_count,
        )

    @property
    def stats(self) -> dict[str, int]:
        return {
            "bytes_sent": self._bytes_sent,
            "partial_count": self._partial_count,
            "committed_count": self._committed_count,
            "avg_capture_to_send_ms": int(
                self._send_latency_ms_total / max(self._send_latency_samples, 1)
            ),
            "messages_received": self._messages_received,
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
                    trace = LatencyTrace(
                        microphone_ts=frame.capture_monotonic,
                        queue_ts=frame.queued_monotonic,
                    )
                    self._last_sent_trace = trace
                    self._capture_debug_audio(frame.data)
                    pcm_buffer.extend(float32_to_pcm16(frame.data))
            except queue.Empty:
                pass

            while len(pcm_buffer) >= _SEND_CHUNK_BYTES:
                chunk = bytes(pcm_buffer[:_SEND_CHUNK_BYTES])
                del pcm_buffer[:_SEND_CHUNK_BYTES]
                await ws.send(chunk)
                send_ts = time.monotonic()
                if self._last_sent_trace is not None:
                    self._last_sent_trace.wlk_send_ts = send_ts
                    if self._last_sent_trace.microphone_ts is not None:
                        self._send_latency_ms_total += (send_ts - self._last_sent_trace.microphone_ts) * 1000.0
                        self._send_latency_samples += 1
                self._bytes_sent += len(chunk)

            await asyncio.sleep(0.010)

        try:
            if pcm_buffer:
                await ws.send(bytes(pcm_buffer))
                self._bytes_sent += len(pcm_buffer)
            await ws.send(b"")
        except ConnectionClosed:
            pass

    async def _recv_loop(self, ws: Any) -> None:
        """Receive WLK JSON messages and route transcript events."""
        async for raw in ws:
            if self._stop_event.is_set():
                break
            if isinstance(raw, bytes):
                logger.info("WLK inbound binary frame after config=%s bytes=%d", self._received_config, len(raw))
                raw = raw.decode("utf-8", errors="replace")
            if self._received_config:
                logger.info("WLK inbound JSON raw: %s", raw)
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("Non-JSON WLK message: %r", raw)
                continue
            self._messages_received += 1
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        """Route WLK native, diff-mode, and legacy transcript events to queues."""
        msg_type = msg.get("type") or msg.get("status") or ""

        if msg_type == "config":
            self._received_config = True
            self._config_received_event.set()
            logger.info(
                "WLK config received — useAudioWorklet=%s mode=%s",
                msg.get("useAudioWorklet"),
                msg.get("mode"),
            )
            return

        if msg_type == "ready_to_stop":
            logger.info("WLK reported ready_to_stop")
            return

        if msg.get("error"):
            logger.warning("WLK websocket error message: %s", msg.get("error"))

        partial = self._extract_partial_text(msg)
        if partial:
            logger.info("WLK parser branch=partial_buffer text=%r", partial[:120])
            self._put_partial(partial)

        committed_lines = self._extract_committed_lines(msg)
        if committed_lines:
            logger.info(
                "WLK parser branch=committed_lines partial=%r new_committed=%d partial_q=%d committed_q=%d",
                partial[:80],
                len(committed_lines),
                self._partial_queue.qsize(),
                self._committed_queue.qsize(),
            )
            for text in committed_lines:
                self._put_committed(text)
            return

        legacy_text = (msg.get("text") or msg.get("transcript") or "").strip()
        if not legacy_text:
            logger.debug("WLK message without transcript text: %r", msg)
            return

        if msg_type in ("partial", "interim", "processing"):
            logger.info("WLK parser branch=legacy_partial text=%r", legacy_text[:120])
            self._put_partial(legacy_text)
            return

        if msg_type in ("committed", "final", "complete", "completed", ""):
            logger.info("WLK parser branch=legacy_committed text=%r", legacy_text[:120])
            self._put_committed(legacy_text)
            return

        logger.debug("Unknown WLK message type %r: %r", msg_type, msg)

    def _extract_partial_text(self, msg: dict[str, Any]) -> str:
        """Return WLK native buffer text, if present."""
        for key in ("buffer_transcription", "buffer_diarization", "buffer_translation"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _extract_committed_lines(self, msg: dict[str, Any]) -> list[str]:
        """Return newly committed WLK line texts, de-duplicating full-state updates."""
        raw_lines: list[Any] = []
        msg_type = msg.get("type")
        if msg_type == "diff":
            raw_lines = list(msg.get("new_lines") or [])
        elif "lines" in msg:
            raw_lines = list(msg.get("lines") or [])

        new_texts: list[str] = []
        for line in raw_lines:
            if not isinstance(line, dict):
                continue
            text = line.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            if line.get("speaker") == -2:
                continue
            key = (
                str(line.get("speaker", "")),
                str(line.get("start", "")),
                str(line.get("end", "")),
                text.strip(),
            )
            if key in self._last_committed_keys:
                continue
            self._last_committed_keys.add(key)
            new_texts.append(text.strip())
        return new_texts

    def _put_partial(self, text: str) -> None:
        try:
            self._partial_queue.put_nowait(text)
            self._partial_count += 1
            logger.info("Queued partial transcript partial_q=%d", self._partial_queue.qsize())
        except queue.Full:
            logger.debug("partial_queue full — partial transcript dropped")

    def _capture_debug_audio(self, audio: np.ndarray) -> None:
        if self._debug_capture_written:
            return
        self._debug_capture_samples.append(audio.copy())
        sample_count = sum(chunk.size for chunk in self._debug_capture_samples)
        if sample_count < int(config.SAMPLE_RATE * _DEBUG_CAPTURE_SECONDS):
            return

        captured = np.concatenate(self._debug_capture_samples)[: int(config.SAMPLE_RATE * _DEBUG_CAPTURE_SECONDS)]
        pcm = float32_to_pcm16(captured)
        with wave.open(str(_DEBUG_CAPTURE_PATH), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(config.SAMPLE_RATE)
            wav_file.writeframes(pcm)
        self._debug_capture_written = True
        self._debug_capture_samples.clear()
        rms = float(np.sqrt(np.mean(captured ** 2)))
        peak = float(np.max(np.abs(captured)))
        logger.info(
            "Wrote %s for microphone verification — duration=%.1fs sample_rate=%d channels=1 format=pcm_s16le rms=%.6f peak=%.6f",
            _DEBUG_CAPTURE_PATH,
            _DEBUG_CAPTURE_SECONDS,
            config.SAMPLE_RATE,
            rms,
            peak,
        )

    def _put_committed(self, text: str) -> None:
        trace = self._last_sent_trace or LatencyTrace()
        trace.transcript_ts = time.monotonic()
        line = CommittedLine(
            text=text,
            timestamp=time.time(),
            latency_trace=trace,
        )
        try:
            self._committed_queue.put_nowait(line)
            self._committed_count += 1
            logger.info("Queued committed transcript committed_q=%d", self._committed_queue.qsize())
        except queue.Full:
            logger.warning("committed_queue full — committed line dropped")

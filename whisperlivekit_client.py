"""Asynchronous WhisperLiveKit PCM client with speaker-aware transcript routing."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
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
from event_models import CommittedLine, LatencyTrace
from queue_fanout import as_queue_list, publish_latest
from speaker_utils import SpeakerRegistry, normalize_speaker_id, wlk_relative_to_wallclock

logger = logging.getLogger(__name__)
_SEND_CHUNK_BYTES = int(config.SAMPLE_RATE * 0.040 * 2)
_RECONNECT_DELAY_SEC = 2.0
_MAX_RECONNECT_ATTEMPTS = 5
_MAX_DEDUP_KEYS = 10_000


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return np.ascontiguousarray((clipped * 32768.0).clip(-32768, 32767).astype("<i2")).tobytes()


class WhisperLiveKitClient:
    """Stream microphone PCM to WLK and fan committed lines out to consumers."""

    def __init__(
        self,
        wlk_queue: queue.Queue[TimestampedFrame],
        partial_queue: queue.Queue[str],
        committed_queue: queue.Queue[CommittedLine] | list[queue.Queue[CommittedLine]],
        url: str = config.WLK_URL,
        speaker_registry: SpeakerRegistry | None = None,
        diarization_enabled: bool = config.ENABLE_SPEAKER_DIARIZATION,
    ) -> None:
        self._wlk_queue = wlk_queue
        self._partial_queue = partial_queue
        self._committed_queues = as_queue_list(committed_queue)
        self._url = url
        self._speaker_registry = speaker_registry or SpeakerRegistry()
        self._diarization_enabled = diarization_enabled

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ws: Any | None = None
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._ready_to_stop_event = threading.Event()
        self._last_error: BaseException | None = None

        self._bytes_sent = 0
        self._frames_consumed = 0
        self._chunks_sent = 0
        self._messages_received = 0
        self._partial_count = 0
        self._committed_count = 0
        self._last_frame_index = 0
        self._last_send_monotonic = 0.0
        self._last_message_type = ""
        self._last_message_text = ""
        self._last_partial_text = ""
        self._latest_speaker_id: int | str | None = None
        self._latest_speaker_label: str | None = None
        self._remaining_time_diarization = 0.0
        self._stream_start_wallclock: float | None = None
        self._last_sent_trace: LatencyTrace | None = None
        self._emitted_line_keys: set[tuple[Any, ...]] = set()
        self._emitted_line_order: deque[tuple[Any, ...]] = deque()

    def start(self) -> None:
        if websockets is None:
            raise ImportError("websockets is not installed. Run: pip install websockets")
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._connected_event.clear()
        self._last_error = None
        self._thread = threading.Thread(target=self._run_event_loop, name="wlk-client", daemon=True)
        self._thread.start()
        logger.info("WhisperLiveKit client starting url=%s", self._url)

    def wait_until_connected(self, timeout: float = 10.0) -> None:
        if self._connected_event.wait(timeout):
            return
        if self._last_error is not None:
            raise RuntimeError(f"WLK websocket failed to connect: {self._last_error}")
        raise TimeoutError(f"WLK websocket did not connect to {self._url} within {timeout:.1f}s")

    def stop(self) -> None:
        self._stop_event.set()
        self._connected_event.clear()
        if self._thread:
            self._thread.join(timeout=7.0)
        if self._thread and self._thread.is_alive() and self._loop and self._loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._finish_and_close(), self._loop)
                future.result(timeout=2.0)
            except Exception:  # noqa: BLE001
                logger.debug("WLK forced close did not complete", exc_info=True)
        self._ws = None
        logger.info(
            "WhisperLiveKit client stopped bytes_sent=%d committed=%d",
            self._bytes_sent,
            self._committed_count,
        )

    async def _finish_and_close(self) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(b"")
        except ConnectionClosed:
            return
        await self._ws.close()

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "connected": self._connected_event.is_set(),
            "queue_depth": self._wlk_queue.qsize(),
            "frames_consumed": self._frames_consumed,
            "last_frame_index": self._last_frame_index,
            "bytes_sent": self._bytes_sent,
            "chunks_sent": self._chunks_sent,
            "last_send_age_sec": (
                round(time.monotonic() - self._last_send_monotonic, 3)
                if self._last_send_monotonic
                else None
            ),
            "messages_received": self._messages_received,
            "partial_count": self._partial_count,
            "committed_count": self._committed_count,
            "last_message_type": self._last_message_type,
            "last_message_text": self._last_message_text,
            "last_error": str(self._last_error) if self._last_error else "",
            "remaining_time_diarization": self._remaining_time_diarization,
            "speakers_seen": list(self._speaker_registry.speakers_seen),
            "latest_speaker_id": self._latest_speaker_id,
            "latest_speaker_label": self._latest_speaker_label,
        }

    @property
    def speaker_registry(self) -> SpeakerRegistry:
        return self._speaker_registry

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
                    logger.error("WLK reconnect limit reached: %s", exc)
                    break
                logger.warning("WLK connection failed; retry %d/%d: %s", attempts, _MAX_RECONNECT_ATTEMPTS, exc)
                await asyncio.sleep(_RECONNECT_DELAY_SEC)

    async def _connect_once(self) -> None:
        assert websockets is not None
        logger.info("Connecting to WLK at %s", self._url)
        async with websockets.connect(self._url) as ws:
            self._ws = ws
            self._reset_protocol_session()
            self._last_error = None
            self._connected_event.set()
            send_task = asyncio.create_task(self._send_loop(ws))
            recv_task = asyncio.create_task(self._recv_loop(ws))
            done, pending = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()

    def _reset_protocol_session(self) -> None:
        self._stream_start_wallclock = None
        self._ready_to_stop_event.clear()
        self._emitted_line_keys.clear()
        self._emitted_line_order.clear()
        self._speaker_registry.reset()
        self._latest_speaker_id = None
        self._latest_speaker_label = None

    async def _send_loop(self, ws: Any) -> None:
        pcm_buffer = bytearray()
        while not self._stop_event.is_set():
            try:
                while True:
                    frame = self._wlk_queue.get_nowait()
                    if self._stream_start_wallclock is None:
                        self._stream_start_wallclock = frame.timestamp
                    self._frames_consumed += 1
                    self._last_frame_index = frame.frame_index
                    self._last_sent_trace = LatencyTrace(
                        microphone_ts=frame.capture_monotonic or None,
                        queue_ts=frame.queued_monotonic or None,
                        transcription_input_ts=time.monotonic(),
                    )
                    pcm_buffer.extend(float32_to_pcm16(frame.data))
            except queue.Empty:
                pass
            while len(pcm_buffer) >= _SEND_CHUNK_BYTES:
                chunk = bytes(pcm_buffer[:_SEND_CHUNK_BYTES])
                del pcm_buffer[:_SEND_CHUNK_BYTES]
                await ws.send(chunk)
                self._bytes_sent += len(chunk)
                self._chunks_sent += 1
                self._last_send_monotonic = time.monotonic()
            await asyncio.sleep(0.010)

        # Microphone has stopped. Drain its bounded queue, flush every pending
        # byte, then use WLK's documented end-of-audio signal so final lines
        # can arrive before the websocket closes.
        try:
            while True:
                frame = self._wlk_queue.get_nowait()
                if self._stream_start_wallclock is None:
                    self._stream_start_wallclock = frame.timestamp
                self._frames_consumed += 1
                self._last_frame_index = frame.frame_index
                pcm_buffer.extend(float32_to_pcm16(frame.data))
        except queue.Empty:
            pass
        try:
            if pcm_buffer:
                await ws.send(bytes(pcm_buffer))
                self._bytes_sent += len(pcm_buffer)
                self._chunks_sent += 1
                self._last_send_monotonic = time.monotonic()
            await ws.send(b"")
            deadline = time.monotonic() + 5.0
            while not self._ready_to_stop_event.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
        except ConnectionClosed:
            pass

    async def _recv_loop(self, ws: Any) -> None:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Ignoring non-JSON WLK message")
                continue
            if not isinstance(msg, dict):
                logger.warning("Ignoring malformed WLK message: %r", msg)
                continue
            self._messages_received += 1
            self._dispatch(msg)

    def _dispatch(self, msg: dict[str, Any]) -> None:
        msg_type = str(msg.get("type") or msg.get("status") or "snapshot")
        self._last_message_type = msg_type
        if msg_type == "ready_to_stop":
            self._ready_to_stop_event.set()
            return
        lag = msg.get("remaining_time_diarization")
        if isinstance(lag, (int, float)):
            self._remaining_time_diarization = float(lag)

        buffer_text = msg.get("buffer_transcription")
        if buffer_text is not None:
            self._set_partial_text(str(buffer_text))

        if msg_type == "diff":
            self._dispatch_lines(msg.get("new_lines", []))
            return
        if "lines" in msg:
            self._dispatch_lines(msg.get("lines", []))
            return

        text = str(msg.get("text") or msg.get("transcript") or "").strip()
        if msg_type in {"partial", "interim", "processing"}:
            self._set_partial_text(text)
        elif msg_type in {"committed", "final", "complete", "completed"} and text:
            self._put_committed_text(text)
        elif msg_type not in {"config", "ready_to_stop", "active_transcription", "no_audio_detected"}:
            logger.debug("Unknown WLK message type %r", msg_type)

    def _dispatch_lines(self, lines: Any) -> None:
        if not isinstance(lines, list):
            logger.warning("Ignoring malformed WLK lines payload")
            return
        for raw_line in lines:
            if not isinstance(raw_line, dict):
                continue
            raw_speaker_id = normalize_speaker_id(raw_line.get("speaker"))
            if raw_speaker_id == -2:
                continue
            speaker_id = raw_speaker_id if self._diarization_enabled else None
            text = str(raw_line.get("text") or "").strip()
            if not text:
                continue
            key = (speaker_id, raw_line.get("start"), raw_line.get("end"), text)
            if not self._remember_line_key(key):
                continue
            start_time = wlk_relative_to_wallclock(self._stream_start_wallclock, raw_line.get("start"))
            end_time = wlk_relative_to_wallclock(self._stream_start_wallclock, raw_line.get("end"))
            label = self._speaker_registry.observe(speaker_id)
            trace = replace(self._last_sent_trace) if self._last_sent_trace else LatencyTrace()
            trace.transcript_ts = time.monotonic()
            line = CommittedLine(
                text=text,
                timestamp=end_time or time.time(),
                latency_trace=trace,
                speaker_id=speaker_id,
                speaker_label=label,
                start_time=start_time,
                end_time=end_time,
            )
            self._publish_committed(line)

    def _remember_line_key(self, key: tuple[Any, ...]) -> bool:
        if key in self._emitted_line_keys:
            return False
        self._emitted_line_keys.add(key)
        self._emitted_line_order.append(key)
        if len(self._emitted_line_order) > _MAX_DEDUP_KEYS:
            self._emitted_line_keys.discard(self._emitted_line_order.popleft())
        return True

    def _put_committed_text(self, text: str) -> None:
        clean = text.strip()
        key = (None, None, None, clean)
        if not clean or not self._remember_line_key(key):
            return
        trace = replace(self._last_sent_trace) if self._last_sent_trace else LatencyTrace()
        trace.transcript_ts = time.monotonic()
        self._publish_committed(CommittedLine(text=clean, timestamp=time.time(), latency_trace=trace))

    def _publish_committed(self, line: CommittedLine) -> None:
        delivered = publish_latest(
            self._committed_queues,
            line,
            logger=logger,
            label="committed transcript queue",
        )
        if delivered:
            self._committed_count += 1
            self._last_message_text = line.text
            self._latest_speaker_id = line.speaker_id
            self._latest_speaker_label = line.speaker_label
            logger.info("SPEAKER speaker=%s label=%r text=%r", line.speaker_id, line.speaker_label, line.text)
            logger.info("DIARIZATION speakers_seen=%d", len(self._speaker_registry.speakers_seen))

    def _set_partial_text(self, text: str) -> None:
        clean = text.strip()
        if not clean or clean == self._last_partial_text:
            return
        self._last_partial_text = clean
        self._last_message_text = clean
        while True:
            try:
                self._partial_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self._partial_queue.put_nowait(clean)
            self._partial_count += 1
        except queue.Full:
            pass

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
_QUEUE_CAPTURE_PATH = Path("queue_capture.wav")
_PRE_SEND_CAPTURE_PATH = Path("pre_send.wav")


def audio_stats(audio: np.ndarray) -> dict[str, float | int | str | tuple[int, ...]]:
    """Return stable diagnostics for a NumPy audio array without mutating it."""
    arr = np.asarray(audio)
    return {
        "object_id": id(audio),
        "dtype": str(arr.dtype),
        "shape": tuple(arr.shape),
        "samples": int(arr.size),
        "rms": float(np.sqrt(np.nanmean(arr.astype(np.float32) ** 2))) if arr.size else 0.0,
        "peak": float(np.nanmax(np.abs(arr))) if arr.size else 0.0,
        "min": float(np.nanmin(arr)) if arr.size else 0.0,
        "max": float(np.nanmax(arr)) if arr.size else 0.0,
    }


def log_audio_stage(stage: str, frame: TimestampedFrame, audio: np.ndarray) -> None:
    """Log one frame at a named audio-pipeline stage."""
    stats = audio_stats(audio)
    logger.info(
        "Audio trace stage=%s frame_id=%d object_id=%s dtype=%s shape=%s samples=%d rms=%.8f peak=%.8f min=%.8f max=%.8f",
        stage,
        frame.frame_index,
        stats["object_id"],
        stats["dtype"],
        stats["shape"],
        stats["samples"],
        stats["rms"],
        stats["peak"],
        stats["min"],
        stats["max"],
    )


def write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int = config.SAMPLE_RATE) -> None:
    """Write already-converted PCM16 bytes to a mono WAV without regenerating samples."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)


def log_wav_report(path: Path) -> None:
    """Report WAV duration and PCM-level statistics for a diagnostic capture."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        payload = wav_file.readframes(frames)
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    duration = frames / sample_rate if sample_rate else 0.0
    logger.info(
        "Diagnostic WAV %s — duration=%.3fs rms=%.8f peak=%.8f sample_rate=%d channels=%d sample_width=%d",
        path,
        duration,
        rms,
        peak,
        sample_rate,
        channels,
        sample_width,
    )


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert mono float32 audio to contiguous signed 16-bit little-endian PCM."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = np.ascontiguousarray((clipped * 32768.0).clip(-32768, 32767).astype("<i2"))
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
        self._queue_capture_samples: list[np.ndarray] = []
        self._queue_capture_written = False
        self._pre_send_capture = bytearray()
        self._pre_send_capture_written = False

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
                    log_audio_stage("queue_retrieval", frame, frame.data)
                    self._capture_queue_audio(frame.data)
                    pcm = float32_to_pcm16(frame.data)
                    self._log_pcm_conversion(frame, frame.data, pcm)
                    pcm_buffer.extend(pcm)
            except queue.Empty:
                pass

            while len(pcm_buffer) >= _SEND_CHUNK_BYTES:
                chunk = bytes(pcm_buffer[:_SEND_CHUNK_BYTES])
                del pcm_buffer[:_SEND_CHUNK_BYTES]
                self._log_websocket_chunk(chunk)
                self._capture_pre_send_audio(chunk)
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
        write_pcm16_wav(_DEBUG_CAPTURE_PATH, pcm)
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
        log_wav_report(_DEBUG_CAPTURE_PATH)

    def _capture_queue_audio(self, audio: np.ndarray) -> None:
        if self._queue_capture_written:
            return
        self._queue_capture_samples.append(audio.copy())
        sample_count = sum(chunk.size for chunk in self._queue_capture_samples)
        if sample_count < int(config.SAMPLE_RATE * _DEBUG_CAPTURE_SECONDS):
            return
        captured = np.concatenate(self._queue_capture_samples)[: int(config.SAMPLE_RATE * _DEBUG_CAPTURE_SECONDS)]
        write_pcm16_wav(_QUEUE_CAPTURE_PATH, float32_to_pcm16(captured))
        self._queue_capture_written = True
        self._queue_capture_samples.clear()
        log_wav_report(_QUEUE_CAPTURE_PATH)

    def _log_websocket_chunk(self, chunk: bytes) -> None:
        pcm_array = np.frombuffer(chunk, dtype="<i2")
        logger.info(
            "Audio trace stage=websocket_send object_id=%s dtype=%s shape=%s samples=%d rms=%.8f peak=%d min=%d max=%d bytes=%d",
            id(chunk),
            pcm_array.dtype,
            pcm_array.shape,
            pcm_array.size,
            float(np.sqrt(np.mean((pcm_array.astype(np.float32) / 32768.0) ** 2))) if pcm_array.size else 0.0,
            int(np.max(np.abs(pcm_array))) if pcm_array.size else 0,
            int(np.min(pcm_array)) if pcm_array.size else 0,
            int(np.max(pcm_array)) if pcm_array.size else 0,
            len(chunk),
        )

    def _capture_pre_send_audio(self, chunk: bytes) -> None:
        if self._pre_send_capture_written:
            return
        needed = int(config.SAMPLE_RATE * _DEBUG_CAPTURE_SECONDS * 2) - len(self._pre_send_capture)
        self._pre_send_capture.extend(chunk[: max(0, needed)])
        if len(self._pre_send_capture) < int(config.SAMPLE_RATE * _DEBUG_CAPTURE_SECONDS * 2):
            return
        write_pcm16_wav(_PRE_SEND_CAPTURE_PATH, bytes(self._pre_send_capture))
        self._pre_send_capture_written = True
        self._pre_send_capture.clear()
        log_wav_report(_PRE_SEND_CAPTURE_PATH)

    def _log_pcm_conversion(self, frame: TimestampedFrame, audio: np.ndarray, pcm: bytes) -> None:
        pcm_array = np.frombuffer(pcm, dtype="<i2")
        logger.info(
            "Audio trace stage=pcm_conversion frame_id=%d float_object_id=%s pcm_object_id=%s dtype=%s shape=%s samples=%d rms=%.8f peak=%d min=%d max=%d",
            frame.frame_index,
            id(audio),
            id(pcm_array),
            pcm_array.dtype,
            pcm_array.shape,
            pcm_array.size,
            float(np.sqrt(np.mean((pcm_array.astype(np.float32) / 32768.0) ** 2))) if pcm_array.size else 0.0,
            int(np.max(np.abs(pcm_array))) if pcm_array.size else 0,
            int(np.min(pcm_array)) if pcm_array.size else 0,
            int(np.max(pcm_array)) if pcm_array.size else 0,
        )
        if frame.frame_index == 1:
            logger.info("First 100 float32 samples frame_id=%d: %s", frame.frame_index, audio[:100].tolist())
            logger.info("First 100 int16 samples frame_id=%d: %s", frame.frame_index, pcm_array[:100].tolist())
            logger.info("First 200 websocket bytes frame_id=%d hex=%s", frame.frame_index, pcm[:200].hex())

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

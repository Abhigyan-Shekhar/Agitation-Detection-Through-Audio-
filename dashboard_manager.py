"""Runtime lifecycle manager for the Streamlit audio dashboard.

This module owns process and thread orchestration for the live recording
pipeline.  The Streamlit app stores one ``DashboardManager`` in
``st.session_state`` and leaves UI rendering in ``dashboard.py``.
"""
from __future__ import annotations

import logging
import os
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

import config
from acoustic_features import AcousticWorker
from audio_pipeline import AudioPipeline
from baseline_manager import BaselineManager
from event_models import CommittedLine, Utterance
from utterance_aggregator import UtteranceAggregator
from whisperlivekit_client import WhisperLiveKitClient

logger = logging.getLogger(__name__)

_WLK_STARTUP_TIMEOUT_SEC = 60.0
_WLK_TCP_POLL_SEC = 0.25
_WLK_TCP_CONNECT_TIMEOUT_SEC = 0.5
_WLK_TERMINATE_TIMEOUT_SEC = 5.0
_WLK_WEBSOCKET_TIMEOUT_SEC = 10.0
_WLK_LOG_TAIL_BYTES = 8_192


class DashboardStartupError(RuntimeError):
    """Raised when the live dashboard runtime cannot start cleanly."""


class DashboardManager:
    """Owns the WLK server, websocket client, audio pipeline, and workers."""

    def __init__(
        self,
        partial_queue: queue.Queue[str],
        committed_queue: queue.Queue[CommittedLine],
        committed_display_queue: queue.Queue[CommittedLine],
        utterance_queue: queue.Queue[Utterance],
        baseline_manager: BaselineManager,
    ) -> None:
        self._partial_queue = partial_queue
        self._committed_queue = committed_queue
        self._committed_display_queue = committed_display_queue
        self._utterance_queue = utterance_queue
        self._baseline_manager = baseline_manager

        self.wlk_proc: subprocess.Popen | None = None
        self.pipeline: AudioPipeline | None = None
        self.wlk_client: WhisperLiveKitClient | None = None
        self.utterance_aggregator: UtteranceAggregator | None = None
        self.acoustic_worker: AcousticWorker | None = None
        self._wlk_log_file: Any | None = None
        self._wlk_log_path: str | None = None
        self._owns_wlk_proc = False

        self._starting = False
        self._stopping = False

    @property
    def is_running(self) -> bool:
        """Return True when the microphone pipeline is active."""
        return self.pipeline is not None and self.pipeline.is_running

    @property
    def server_running(self) -> bool:
        """Return True when this manager owns a live WLK subprocess."""
        return self.wlk_proc is not None and self.wlk_proc.poll() is None

    @property
    def wlk_command(self) -> list[str]:
        """Return the supported WhisperLiveKit command for this dashboard."""
        return [
            sys.executable,
            "-m",
            "whisperlivekit.basic_server",
            "--backend",
            config.WLK_BACKEND,
            "--model",
            config.WLK_MODEL,
            "--language",
            config.WLK_LANGUAGE,
            "--pcm-input",
            "--host",
            config.WLK_HOST,
            "--port",
            str(config.WLK_PORT),
        ]

    def start(self) -> None:
        """Start WLK, connect websocket, then start microphone capture."""
        if self.is_running:
            logger.info("Start requested while dashboard runtime is already running")
            return
        if self._starting:
            logger.info("Start requested while dashboard runtime is already starting")
            return

        logger.info("Starting dashboard runtime")
        self._starting = True
        try:
            self._clear_runtime_queues()
            self._discard_exited_server()
            self._start_wlk()
            self._wait_for_tcp_ready()
            self._create_pipeline_components()
            self._connect_websocket()
            self._start_workers()
            self._start_microphone()
            logger.info("Dashboard runtime started")
        except Exception as exc:
            logger.exception("Dashboard runtime startup failed")
            self.stop()
            if isinstance(exc, DashboardStartupError):
                raise
            raise DashboardStartupError(f"{self._diagnostics()}\nStartup error: {exc}") from exc
        finally:
            self._starting = False

    def stop(self) -> None:
        """Stop microphone, websocket, workers, and the WLK subprocess."""
        if self._stopping:
            return

        self._stopping = True
        logger.info("Stopping dashboard runtime")
        try:
            logger.info("Stopping microphone")
            self._safe_stop("pipeline", self.pipeline)
            self.pipeline = None

            logger.info("Closing websocket")
            self._safe_stop("wlk_client", self.wlk_client)
            self.wlk_client = None

            logger.info("Stopping workers")
            self._safe_stop("utterance_aggregator", self.utterance_aggregator)
            self.utterance_aggregator = None
            self._safe_stop("acoustic_worker", self.acoustic_worker)
            self.acoustic_worker = None

            logger.info("Terminating WLK")
            self._stop_wlk()
            logger.info("Shutdown complete")
        finally:
            self._stopping = False

    def _start_wlk(self) -> None:
        if not config.WLK_AUTO_LAUNCH:
            logger.info("WLK auto-launch disabled; expecting external server")
            self._owns_wlk_proc = False
            return
        if self.server_running:
            return

        cmd = self.wlk_command
        logger.info("Launching WLK: %s", self._command_text(cmd))
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self._close_wlk_log()
        self._wlk_log_file = tempfile.NamedTemporaryFile(
            prefix="odu-wlk-", suffix=".log", delete=False
        )
        self._wlk_log_path = self._wlk_log_file.name
        try:
            self.wlk_proc = subprocess.Popen(
                cmd,
                stdout=self._wlk_log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            self._owns_wlk_proc = True
        except Exception as exc:  # noqa: BLE001
            self._close_wlk_log()
            raise DashboardStartupError(
                f"{self._diagnostics()}\nException: {exc}"
            ) from exc

    def _wait_for_tcp_ready(self) -> None:
        logger.info("Waiting for TCP %s:%s", config.WLK_HOST, config.WLK_PORT)
        deadline = time.monotonic() + _WLK_STARTUP_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self.wlk_proc is not None and self.wlk_proc.poll() is not None:
                raise DashboardStartupError(self._exit_diagnostics())
            try:
                with socket.create_connection(
                    (config.WLK_HOST, config.WLK_PORT),
                    timeout=_WLK_TCP_CONNECT_TIMEOUT_SEC,
                ):
                    logger.info("TCP ready")
                    return
            except OSError:
                time.sleep(_WLK_TCP_POLL_SEC)

        raise DashboardStartupError(
            f"{self._diagnostics()}\nTimed out waiting for TCP readiness "
            f"after {_WLK_STARTUP_TIMEOUT_SEC:.0f}s."
        )

    def _create_pipeline_components(self) -> None:
        self.pipeline = AudioPipeline()
        self.wlk_client = WhisperLiveKitClient(
            wlk_queue=self.pipeline.wlk_queue,
            partial_queue=self._partial_queue,
            committed_queue=[
                self._committed_queue,
                self._committed_display_queue,
            ],
        )
        self.utterance_aggregator = UtteranceAggregator(
            committed_queue=self._committed_queue,
            utterance_queue=self._utterance_queue,
        )
        self.acoustic_worker = AcousticWorker(
            acoustic_queue=self.pipeline.acoustic_queue,
        )
        self._patch_acoustic_worker_baseline_feed()

    def _connect_websocket(self) -> None:
        if self.wlk_client is None:
            raise DashboardStartupError(f"{self._diagnostics()}\nNo WLK client exists.")
        logger.info("Connecting websocket")
        self.wlk_client.start()
        self.wlk_client.wait_until_connected(timeout=_WLK_WEBSOCKET_TIMEOUT_SEC)
        logger.info("Websocket connected")

    def _start_workers(self) -> None:
        if self.utterance_aggregator is None or self.acoustic_worker is None:
            raise DashboardStartupError(f"{self._diagnostics()}\nWorkers are not ready.")
        self.utterance_aggregator.start()
        self.acoustic_worker.start()

    def _start_microphone(self) -> None:
        if self.pipeline is None:
            raise DashboardStartupError(f"{self._diagnostics()}\nAudioPipeline is not ready.")
        logger.info("Starting microphone")
        self.pipeline.start()

    def _patch_acoustic_worker_baseline_feed(self) -> None:
        """Preserve the existing dashboard baseline feed behavior."""
        acoustic_worker = self.acoustic_worker
        if acoustic_worker is None:
            return

        def _run_with_baseline_feed() -> None:
            while not acoustic_worker._stop_event.is_set():
                acoustic_worker._drain_queue()
                now = time.time()
                if now - acoustic_worker._last_extraction_time >= acoustic_worker._hop_sec:
                    records = acoustic_worker._ring.latest_window(
                        acoustic_worker._window_sec
                    )
                    if records:
                        window_end = now
                        window_start = window_end - acoustic_worker._window_sec
                        feat = acoustic_worker._extractor.extract(
                            records, window_start, window_end
                        )
                        with acoustic_worker._lock:
                            acoustic_worker._windows.append(feat)
                        acoustic_worker._last_extraction_time = now
                        acoustic_worker._windows_extracted += 1
                        self._baseline_manager.feed(feat)
                time.sleep(0.010)

        acoustic_worker._thread = threading.Thread(
            target=_run_with_baseline_feed,
            name="acoustic-worker",
            daemon=True,
        )

    def _stop_wlk(self) -> None:
        proc = self.wlk_proc
        if proc is None:
            self._close_wlk_log()
            return

        if not self._owns_wlk_proc:
            self.wlk_proc = None
            self._close_wlk_log()
            return

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_WLK_TERMINATE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_WLK_TERMINATE_TIMEOUT_SEC)
        self.wlk_proc = None
        self._owns_wlk_proc = False
        self._close_wlk_log()

    def _discard_exited_server(self) -> None:
        if self.wlk_proc is not None and self.wlk_proc.poll() is not None:
            self.wlk_proc = None

    def _clear_runtime_queues(self) -> None:
        for q in (
            self._partial_queue,
            self._committed_queue,
            self._committed_display_queue,
            self._utterance_queue,
        ):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def _safe_stop(self, name: str, obj: Any) -> None:
        if obj is None:
            return
        try:
            obj.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping %s", name)

    def _diagnostics(self) -> str:
        return "\n".join([
            "WhisperLiveKit startup failed.",
            f"Launch command: {self._command_text(self.wlk_command)}",
            f"WebSocket URL: {config.WLK_URL}",
            f"Host: {config.WLK_HOST}",
            f"Port: {config.WLK_PORT}",
            f"Backend: {config.WLK_BACKEND}",
            f"Model: {config.WLK_MODEL}",
            f"WLK log: {self._wlk_log_path or '(not started)'}",
        ])

    def _exit_diagnostics(self) -> str:
        if self.wlk_proc is None:
            return self._diagnostics()
        self.wlk_proc.wait(timeout=_WLK_TERMINATE_TIMEOUT_SEC)
        return "\n".join([
            self._diagnostics(),
            f"Exit code: {self.wlk_proc.returncode}",
            f"WLK log tail:\n{self._read_wlk_log_tail()}",
        ])

    def _read_wlk_log_tail(self) -> str:
        if not self._wlk_log_path or not os.path.exists(self._wlk_log_path):
            return "(empty)"
        if self._wlk_log_file is not None:
            self._wlk_log_file.flush()
        with open(self._wlk_log_path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - _WLK_LOG_TAIL_BYTES))
            return log_file.read().decode(errors="replace").strip() or "(empty)"

    def _close_wlk_log(self) -> None:
        if self._wlk_log_file is None:
            return
        try:
            self._wlk_log_file.close()
        finally:
            self._wlk_log_file = None

    @staticmethod
    def _command_text(cmd: list[str]) -> str:
        return subprocess.list2cmdline(cmd)

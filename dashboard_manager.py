"""Lifecycle manager for local transcription, WLK, audio, and analysis workers."""
from __future__ import annotations

import importlib.util
import logging
import os
import platform
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import http.client
from typing import Any

import config
from acoustic_features import AcousticWorker
from audio_pipeline import AudioPipeline
from baseline_manager import BaselineManager
from event_models import AcousticFeatureWindow, CommittedLine, Utterance
from transcriber import TranscriptionWorker
from utterance_aggregator import UtteranceAggregator
from whisperlivekit_client import WhisperLiveKitClient


logger = logging.getLogger(__name__)
_TCP_POLL_SEC = 0.25
_TCP_TIMEOUT_SEC = 0.5
_WS_TIMEOUT_SEC = 10.0
_TERMINATE_TIMEOUT_SEC = 5.0
_LOG_TAIL_BYTES = 8192
_OWNED_WLK_PROC: subprocess.Popen[bytes] | None = None
_OWNED_WLK_LOCK = threading.Lock()


class DashboardStartupError(RuntimeError):
    """Raised when the live dashboard runtime cannot start cleanly."""


class DashboardManager:
    """Own exactly one transcription path plus the shared acoustic pipeline."""

    def __init__(
        self,
        partial_queue: queue.Queue[str],
        committed_queue: queue.Queue[CommittedLine],
        utterance_queue: queue.Queue[Utterance],
        baseline_manager: BaselineManager,
        committed_display_queue: queue.Queue[CommittedLine] | None = None,
    ) -> None:
        self._partial_queue = partial_queue
        self._committed_queue = committed_queue
        self._committed_display_queue = committed_display_queue or queue.Queue(maxsize=100)
        self._utterance_queue = utterance_queue
        self._baseline_manager = baseline_manager

        self.pipeline: AudioPipeline | None = None
        self.transcription_worker: TranscriptionWorker | None = None
        self.wlk_client: WhisperLiveKitClient | None = None
        self.wlk_proc: subprocess.Popen[bytes] | None = None
        self.utterance_aggregator: UtteranceAggregator | None = None
        self.acoustic_worker: AcousticWorker | None = None
        self._wlk_log_file: Any | None = None
        self._wlk_log_path: str | None = None
        self._owns_wlk_proc = False
        self._starting = False
        self._stopping = False

    @property
    def uses_wlk(self) -> bool:
        return config.TRANSCRIPTION_ENGINE == "whisperlivekit"

    @property
    def is_running(self) -> bool:
        return self.pipeline is not None and self.pipeline.is_running

    @property
    def server_running(self) -> bool:
        return self.wlk_proc is not None and self.wlk_proc.poll() is None

    @property
    def wlk_command(self) -> list[str]:
        command = [
            sys.executable,
            "-u",
            "-m",
            "whisperlivekit.basic_server",
            "--backend",
            config.WLK_BACKEND,
            "--model",
            config.WHISPER_MODEL,
            "--language",
            config.WHISPER_LANGUAGE or "auto",
            "--pcm-input",
            "--host",
            config.WLK_HOST,
            "--port",
            str(config.WLK_PORT),
        ]
        if config.ENABLE_SPEAKER_DIARIZATION:
            command.extend(["--diarization", "--diarization-backend", config.DIARIZATION_BACKEND])
        return command

    def start(self) -> None:
        if self.is_running or self._starting:
            return
        self._starting = True
        try:
            self._clear_runtime_queues()
            if self.uses_wlk:
                self._start_wlk()
                self._wait_for_wlk_ready()
            self._create_pipeline_components()
            self._start_workers()
            self._start_microphone()
            logger.info(
                "Dashboard runtime started engine=%s diarization=%s",
                config.TRANSCRIPTION_ENGINE,
                config.ENABLE_SPEAKER_DIARIZATION and self.uses_wlk,
            )
        except Exception as exc:
            logger.exception("Dashboard runtime startup failed")
            self.stop()
            if isinstance(exc, DashboardStartupError):
                raise
            raise DashboardStartupError(f"{self._diagnostics()}\nStartup error: {exc}") from exc
        finally:
            self._starting = False

    def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        try:
            self._safe_stop("pipeline", self.pipeline)
            self.pipeline = None
            self._safe_stop("wlk_client", self.wlk_client)
            self.wlk_client = None
            self._safe_stop("transcription_worker", self.transcription_worker)
            self.transcription_worker = None
            self._safe_stop("utterance_aggregator", self.utterance_aggregator)
            self.utterance_aggregator = None
            self._safe_stop("acoustic_worker", self.acoustic_worker)
            self.acoustic_worker = None
            self._stop_wlk()
        finally:
            self._stopping = False

    def _create_pipeline_components(self) -> None:
        self.pipeline = AudioPipeline()
        committed_targets = [self._committed_queue, self._committed_display_queue]
        if self.uses_wlk:
            self.wlk_client = WhisperLiveKitClient(
                wlk_queue=self.pipeline.transcription_queue,
                partial_queue=self._partial_queue,
                committed_queue=committed_targets,
            )
        else:
            self.transcription_worker = TranscriptionWorker(
                audio_queue=self.pipeline.transcription_queue,
                partial_queue=self._partial_queue,
                committed_queue=committed_targets,
            )
        self.utterance_aggregator = UtteranceAggregator(
            committed_queue=self._committed_queue,
            utterance_queue=self._utterance_queue,
        )
        self.acoustic_worker = AcousticWorker(
            acoustic_queue=self.pipeline.acoustic_queue,
            on_window=self._handle_acoustic_window,
        )

    def _start_workers(self) -> None:
        if self.utterance_aggregator is None or self.acoustic_worker is None:
            raise DashboardStartupError(f"{self._diagnostics()}\nAnalysis workers are not ready.")
        if self.uses_wlk:
            if self.wlk_client is None:
                raise DashboardStartupError(f"{self._diagnostics()}\nWLK client is not ready.")
            self.wlk_client.start()
            self.wlk_client.wait_until_connected(timeout=_WS_TIMEOUT_SEC)
        else:
            if self.transcription_worker is None:
                raise DashboardStartupError(f"{self._diagnostics()}\nTranscriber is not ready.")
            self.transcription_worker.start()
        self.utterance_aggregator.start()
        self.acoustic_worker.start()

    def _start_microphone(self) -> None:
        if self.pipeline is None:
            raise DashboardStartupError(f"{self._diagnostics()}\nAudio pipeline is not ready.")
        self.pipeline.start()

    def _handle_acoustic_window(self, feat: AcousticFeatureWindow) -> None:
        """Feed extracted acoustic windows into the shared baseline manager."""
        before = self._baseline_manager.calibration_window_count
        was_calibrating = self._baseline_manager.is_calibrating
        self._baseline_manager.feed(feat)
        after = self._baseline_manager.calibration_window_count
        if was_calibrating and after != before:
            logger.info(
                "Baseline calibration collected acoustic window %d/%d "
                "(manager_id=%s)",
                after,
                self._baseline_manager.minimum_windows_for_personal,
                id(self._baseline_manager),
            )

    def _safe_stop(self, name: str, obj: Any) -> None:
        if obj is None:
            return
        try:
            obj.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping %s", name)

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

    def _start_wlk(self) -> None:
        """Ensure a single WhisperLiveKit server is available for this manager."""
        self._validate_wlk_runtime()
        if not self.uses_wlk:
            return
        if self.server_running:
            return

        global _OWNED_WLK_PROC
        with _OWNED_WLK_LOCK:
            if _OWNED_WLK_PROC is not None and _OWNED_WLK_PROC.poll() is None:
                self.wlk_proc = _OWNED_WLK_PROC
                self._owns_wlk_proc = True
                logger.info("Reusing dashboard-owned WLK process pid=%s", self.wlk_proc.pid)
                return

            if self._is_tcp_open():
                self.wlk_proc = None
                self._owns_wlk_proc = False
                logger.info("Using existing WLK server at %s:%d", config.WLK_HOST, config.WLK_PORT)
                return

            if not config.WLK_AUTO_LAUNCH:
                self.wlk_proc = None
                self._owns_wlk_proc = False
                logger.info("WLK auto-launch disabled; expecting external server at %s", config.WLK_URL)
                return

            self._open_wlk_log()
            logger.info("Starting WhisperLiveKit: %s", subprocess.list2cmdline(self.wlk_command))
            self.wlk_proc = subprocess.Popen(
                self.wlk_command,
                stdout=self._wlk_log_file,
                stderr=subprocess.STDOUT,
            )
            self._owns_wlk_proc = True
            _OWNED_WLK_PROC = self.wlk_proc

    def _validate_wlk_runtime(self) -> None:
        if not config.ENABLE_SPEAKER_DIARIZATION:
            return
        if config.DIARIZATION_BACKEND != "sortformer":
            return
        if platform.system() == "Darwin" or sys.version_info >= (3, 13):
            raise DashboardStartupError(
                "WhisperLiveKit Sortformer diarization requires Linux with Python 3.11-3.12. "
                "Set ENABLE_SPEAKER_DIARIZATION=false to use WLK transcription without local Sortformer."
            )
        if importlib.util.find_spec("nemo") is None:
            raise DashboardStartupError(
                "WhisperLiveKit Sortformer diarization requires NeMo. "
                "Install requirements-diarization.txt or set ENABLE_SPEAKER_DIARIZATION=false."
            )

    def _open_wlk_log(self) -> None:
        if self._wlk_log_file is not None:
            return
        log = tempfile.NamedTemporaryFile(prefix="wlk-", suffix=".log", delete=False)
        self._wlk_log_file = log
        self._wlk_log_path = log.name

    def _wait_for_wlk_ready(self) -> None:
        deadline = time.monotonic() + config.WLK_STARTUP_TIMEOUT_SEC
        last_error = ""
        while time.monotonic() < deadline:
            if self.wlk_proc is not None and self.wlk_proc.poll() is not None:
                raise DashboardStartupError(self._exit_diagnostics())
            if self._health_ready():
                logger.info("WhisperLiveKit health check ready at http://%s:%d/health", config.WLK_HOST, config.WLK_PORT)
                return
            if self._is_tcp_open():
                last_error = "TCP port is open but /health is not ready yet"
            else:
                last_error = "TCP port is not open yet"
            time.sleep(_TCP_POLL_SEC)
        raise DashboardStartupError(
            f"{self._diagnostics()}\nWLK readiness timed out after {config.WLK_STARTUP_TIMEOUT_SEC:.1f}s: {last_error}\n"
            f"WLK log tail:\n{self._read_wlk_log_tail()}"
        )

    def _health_ready(self) -> bool:
        try:
            conn = http.client.HTTPConnection(config.WLK_HOST, config.WLK_PORT, timeout=_TCP_TIMEOUT_SEC)
            conn.request("GET", "/health")
            response = conn.getresponse()
            response.read()
            return 200 <= response.status < 300
        except OSError:
            return False
        finally:
            try:
                conn.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass

    def _is_tcp_open(self) -> bool:
        try:
            with socket.create_connection((config.WLK_HOST, config.WLK_PORT), timeout=_TCP_TIMEOUT_SEC):
                return True
        except OSError:
            return False

    def _stop_wlk(self) -> None:
        global _OWNED_WLK_PROC
        proc = self.wlk_proc
        owns_proc = self._owns_wlk_proc
        self.wlk_proc = None
        self._owns_wlk_proc = False

        if proc is not None and owns_proc:
            if proc.poll() is None:
                logger.info("Stopping dashboard-owned WLK process pid=%s", proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=_TERMINATE_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    logger.warning("WLK process did not terminate; killing pid=%s", proc.pid)
                    proc.kill()
                    proc.wait(timeout=_TERMINATE_TIMEOUT_SEC)
            with _OWNED_WLK_LOCK:
                if _OWNED_WLK_PROC is proc:
                    _OWNED_WLK_PROC = None

        if self._wlk_log_file is not None:
            try:
                self._wlk_log_file.close()
            except Exception:  # noqa: BLE001
                logger.debug("Could not close WLK log file", exc_info=True)
            finally:
                self._wlk_log_file = None

    def _diagnostics(self) -> str:
        if not self.uses_wlk:
            return "Local faster-whisper dashboard startup failed."
        return "\n".join([
            "WhisperLiveKit startup failed.",
            f"Launch command: {subprocess.list2cmdline(self.wlk_command)}",
            f"WebSocket URL: {config.WLK_URL}",
            f"Diarization enabled: {config.ENABLE_SPEAKER_DIARIZATION}",
            f"Diarization backend: {config.DIARIZATION_BACKEND}",
            f"WLK log: {self._wlk_log_path or '(not started)'}",
        ])

    def _exit_diagnostics(self) -> str:
        return f"{self._diagnostics()}\nExit code: {self.wlk_proc.returncode if self.wlk_proc else 'unknown'}\nWLK log tail:\n{self._read_wlk_log_tail()}"

    def _read_wlk_log_tail(self) -> str:
        if not self._wlk_log_path or not os.path.exists(self._wlk_log_path):
            return "(empty)"
        if self._wlk_log_file is not None:
            self._wlk_log_file.flush()
        with open(self._wlk_log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _LOG_TAIL_BYTES))
            return handle.read().decode(errors="replace").strip() or "(empty)"

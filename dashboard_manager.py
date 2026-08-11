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
from typing import Any

import config
from acoustic_features import AcousticWorker
from audio_pipeline import AudioPipeline
from baseline_manager import BaselineManager
from event_models import AcousticFeatureWindow, CommittedLine, Utterance
from utterance_aggregator import UtteranceAggregator
from transcriber import TranscriptionWorker
from utterance_aggregator import UtteranceAggregator


logger = logging.getLogger(__name__)
_TCP_POLL_SEC = 0.25
_TCP_TIMEOUT_SEC = 0.5
_WS_TIMEOUT_SEC = 10.0
_TERMINATE_TIMEOUT_SEC = 5.0
_LOG_TAIL_BYTES = 8192


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
                self._wait_for_tcp_ready()
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
<<<<<<< HEAD
        self.acoustic_worker = AcousticWorker(
            acoustic_queue=self.pipeline.acoustic_queue,
            on_window=self._handle_acoustic_window,
        )
=======
        self.acoustic_worker = AcousticWorker(acoustic_queue=self.pipeline.acoustic_queue)
        self._patch_acoustic_worker_baseline_feed()
>>>>>>> origin/codex/add-strange-human-noise-detection

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

<<<<<<< HEAD
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
=======
    def _start_wlk(self) -> None:
        if not config.WLK_AUTO_LAUNCH:
            self._owns_wlk_proc = False
            return
        if (
            config.ENABLE_SPEAKER_DIARIZATION
            and config.DIARIZATION_BACKEND == "sortformer"
            and (platform.system() == "Darwin" or sys.version_info >= (3, 13))
        ):
            raise DashboardStartupError(
                f"{self._diagnostics()}\nThe WLK 0.2.x NeMo/Sortformer extra is supported "
                "by this project on Linux with Python 3.11-3.12. Use the documented "
                "single-speaker fallback here or connect to an external supported WLK server."
            )
        if importlib.util.find_spec("whisperlivekit") is None:
            extra = "[diarization-sortformer]" if config.ENABLE_SPEAKER_DIARIZATION else ""
            raise DashboardStartupError(
                f"{self._diagnostics()}\nWhisperLiveKit is not installed in {sys.executable}. "
                f"Install with: python -m pip install 'whisperlivekit{extra}>=0.2.23,<0.3'"
            )
        self._wlk_log_file = tempfile.NamedTemporaryFile(prefix="odu-wlk-", suffix=".log", delete=False)
        self._wlk_log_path = self._wlk_log_file.name
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.wlk_proc = subprocess.Popen(
                self.wlk_command,
                stdout=self._wlk_log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
        except Exception as exc:  # noqa: BLE001
            raise DashboardStartupError(f"{self._diagnostics()}\nCould not launch WLK: {exc}") from exc
        self._owns_wlk_proc = True

    def _wait_for_tcp_ready(self) -> None:
        deadline = time.monotonic() + config.WLK_STARTUP_TIMEOUT_SEC
        while time.monotonic() < deadline:
            if self.wlk_proc is not None and self.wlk_proc.poll() is not None:
                raise DashboardStartupError(self._exit_diagnostics())
            try:
                with socket.create_connection((config.WLK_HOST, config.WLK_PORT), timeout=_TCP_TIMEOUT_SEC):
                    return
            except OSError:
                time.sleep(_TCP_POLL_SEC)
        raise DashboardStartupError(
            f"{self._diagnostics()}\nTimed out waiting {config.WLK_STARTUP_TIMEOUT_SEC:.0f}s for WLK."
        )

    def _stop_wlk(self) -> None:
        proc = self.wlk_proc
        if proc is not None and self._owns_wlk_proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=_TERMINATE_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=_TERMINATE_TIMEOUT_SEC)
        self.wlk_proc = None
        self._owns_wlk_proc = False
        if self._wlk_log_file is not None:
            self._wlk_log_file.close()
            self._wlk_log_file = None

    def _clear_runtime_queues(self) -> None:
        for target in (
            self._partial_queue,
            self._committed_queue,
            self._committed_display_queue,
            self._utterance_queue,
        ):
            while True:
                try:
                    target.get_nowait()
                except queue.Empty:
                    break

    def _patch_acoustic_worker_baseline_feed(self) -> None:
        acoustic_worker = self.acoustic_worker
        if acoustic_worker is None:
            return

        def _run_with_baseline_feed() -> None:
            while not acoustic_worker._stop_event.is_set():
                acoustic_worker._drain_queue()
                now = time.time()
                if now - acoustic_worker._last_extraction_time >= acoustic_worker._hop_sec:
                    records = acoustic_worker._ring.latest_window(acoustic_worker._window_sec)
                    if records:
                        feat = acoustic_worker._extractor.extract(
                            records,
                            now - acoustic_worker._window_sec,
                            now,
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
>>>>>>> origin/codex/add-strange-human-noise-detection

    def _safe_stop(self, name: str, obj: Any) -> None:
        if obj is None:
            return
        try:
            obj.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping %s", name)

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

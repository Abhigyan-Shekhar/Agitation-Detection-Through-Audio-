"""Runtime lifecycle manager for the Streamlit audio dashboard.

This module owns local process and thread orchestration for the live recording
pipeline.  The Streamlit app stores one ``DashboardManager`` in
``st.session_state`` and leaves UI rendering in ``dashboard.py``.
"""
from __future__ import annotations

import logging
import queue
from typing import Any

import config
from acoustic_features import AcousticWorker
from audio_pipeline import AudioPipeline
from baseline_manager import BaselineManager
from event_models import AcousticFeatureWindow, CommittedLine, Utterance
from utterance_aggregator import UtteranceAggregator
from transcriber import TranscriptionWorker

logger = logging.getLogger(__name__)


class DashboardStartupError(RuntimeError):
    """Raised when the live dashboard runtime cannot start cleanly."""


class DashboardManager:
    """Owns the audio pipeline, local transcriber, and analysis workers."""

    def __init__(
        self,
        partial_queue: queue.Queue[str],
        committed_queue: queue.Queue[CommittedLine],
        utterance_queue: queue.Queue[Utterance],
        baseline_manager: BaselineManager,
    ) -> None:
        self._partial_queue = partial_queue
        self._committed_queue = committed_queue
        self._utterance_queue = utterance_queue
        self._baseline_manager = baseline_manager

        self.pipeline: AudioPipeline | None = None
        self.transcription_worker: TranscriptionWorker | None = None
        self.utterance_aggregator: UtteranceAggregator | None = None
        self.acoustic_worker: AcousticWorker | None = None

        self._starting = False
        self._stopping = False

    @property
    def is_running(self) -> bool:
        """Return True when the microphone pipeline is active."""
        return self.pipeline is not None and self.pipeline.is_running

    def start(self) -> None:
        """Start local workers, then start microphone capture."""
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
            self._create_pipeline_components()
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
        """Stop microphone and local worker threads."""
        if self._stopping:
            return

        self._stopping = True
        logger.info("Stopping dashboard runtime")
        try:
            logger.info("Stopping microphone")
            self._safe_stop("pipeline", self.pipeline)
            self.pipeline = None

            logger.info("Stopping workers")
            self._safe_stop("transcription_worker", self.transcription_worker)
            self.transcription_worker = None
            self._safe_stop("utterance_aggregator", self.utterance_aggregator)
            self.utterance_aggregator = None
            self._safe_stop("acoustic_worker", self.acoustic_worker)
            self.acoustic_worker = None
            logger.info("Shutdown complete")
        finally:
            self._stopping = False

    def _create_pipeline_components(self) -> None:
        self.pipeline = AudioPipeline()
        self.transcription_worker = TranscriptionWorker(
            audio_queue=self.pipeline.transcription_queue,
            partial_queue=self._partial_queue,
            committed_queue=self._committed_queue,
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
        if self.utterance_aggregator is None or self.acoustic_worker is None or self.transcription_worker is None:
            raise DashboardStartupError(f"{self._diagnostics()}\nWorkers are not ready.")
        self.transcription_worker.start()
        self.utterance_aggregator.start()
        self.acoustic_worker.start()

    def _start_microphone(self) -> None:
        if self.pipeline is None:
            raise DashboardStartupError(f"{self._diagnostics()}\nAudioPipeline is not ready.")
        logger.info("Starting microphone")
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

    def _clear_runtime_queues(self) -> None:
        for q in (self._partial_queue, self._committed_queue, self._utterance_queue):
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
        return "Local faster-whisper dashboard startup failed."

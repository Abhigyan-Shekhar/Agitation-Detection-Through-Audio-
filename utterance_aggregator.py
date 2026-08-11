"""Utterance aggregator.

Responsibilities
----------------
* Read ``CommittedLine`` objects from ``committed_queue``.
* Accumulate lines into an in-progress utterance.
* Finalise (emit) the utterance when any condition is met:
    1. No new committed line for ``UTTERANCE_SILENCE_SEC`` seconds.
    2. Accumulated duration reaches ``MAX_UTTERANCE_SEC``.
    3. ``flush()`` is called explicitly (e.g. microphone stopped).
    4. The last committed line ends with terminal punctuation AND at
       least ``MIN_PUNCT_PAUSE_SEC`` has elapsed without a new line.
* Put completed ``Utterance`` objects on ``utterance_queue`` for the
  fusion pipeline to pick up.

Design notes
------------
* Runs in a dedicated background thread (``start()`` / ``stop()``).
* Does not call any ML model — purely time-based heuristics.
* Duplicate committed lines (same text emitted twice by transcriber) are
  de-duplicated via a simple last-line equality check.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import config
from event_models import CommittedLine, Utterance

logger = logging.getLogger(__name__)

# Minimum pause after terminal punctuation to auto-finalise
_MIN_PUNCT_PAUSE_SEC: float = 0.8
_TERMINAL_PUNCT: frozenset[str] = frozenset(".?!")
_POLL_INTERVAL_SEC: float = 0.05    # how often the loop checks timing


class UtteranceAggregator:
    """Assembles committed transcript lines into complete utterances.

    Parameters
    ----------
    committed_queue:
        Source of ``CommittedLine`` objects (from TranscriptionWorker).
    utterance_queue:
        Destination for completed ``Utterance`` objects (to fusion pipeline).
    silence_sec:
        Silence gap that triggers utterance finalisation.
    max_utterance_sec:
        Hard cap on utterance duration.
    """

    def __init__(
        self,
        committed_queue: queue.Queue[CommittedLine],
        utterance_queue: queue.Queue[Utterance],
        silence_sec: float = config.UTTERANCE_SILENCE_SEC,
        max_utterance_sec: float = config.MAX_UTTERANCE_SEC,
    ) -> None:
        self._committed_queue = committed_queue
        self._utterance_queue = utterance_queue
        self._silence_sec = silence_sec
        self._max_utterance_sec = max_utterance_sec

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # In-progress accumulation state
        self._lines: list[CommittedLine] = []
        self._utterance_start: float | None = None
        self._utterance_started_at: float | None = None
        self._last_line_time: float | None = None
        self._last_line_text: str = ""   # for de-duplication
        self._last_line_speaker: int | str | None = None
        self._last_line_source_time: float | None = None
        self._current_speaker_id: int | str | None = None
        self._current_speaker_label: str | None = None

        self._emitted_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the aggregation thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="utterance-aggregator", daemon=True
        )
        self._thread.start()
        logger.info("UtteranceAggregator started")

    def stop(self) -> None:
        """Flush any pending lines and stop the thread."""
        self._stop_event.set()
        self._drain_committed_queue()
        self.flush()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info(
            "UtteranceAggregator stopped — total utterances emitted: %d",
            self._emitted_count,
        )

    def flush(self) -> None:
        """Force-emit the current in-progress utterance immediately."""
        if self._lines:
            self._emit()

    @property
    def emitted_count(self) -> int:
        return self._emitted_count

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # Pull all available lines without blocking
            self._drain_committed_queue()

            # Check finalisation conditions
            if self._lines and self._last_line_time is not None:
                now = time.time()
                elapsed_since_last = now - self._last_line_time
                utterance_age = now - (self._utterance_started_at or now)

                # Condition 1: silence timeout
                if elapsed_since_last >= self._silence_sec:
                    logger.info(
                        "Utterance finalised by silence timeout (%.2fs)",
                        elapsed_since_last,
                    )
                    self._emit()
                    continue

                # Condition 2: max duration hard cap
                if utterance_age >= self._max_utterance_sec:
                    logger.info(
                        "Utterance finalised by max duration (%.2fs)",
                        utterance_age,
                    )
                    self._emit()
                    continue

                # Condition 3: terminal punctuation + short pause
                last_text = self._lines[-1].text.strip()
                if (
                    last_text
                    and last_text[-1] in _TERMINAL_PUNCT
                    and elapsed_since_last >= _MIN_PUNCT_PAUSE_SEC
                ):
                    logger.info(
                        "Utterance finalised by terminal punctuation + pause"
                    )
                    self._emit()
                    continue

            time.sleep(_POLL_INTERVAL_SEC)

    def _drain_committed_queue(self) -> None:
        """Consume all currently available committed lines."""
        while True:
            try:
                line: CommittedLine = self._committed_queue.get_nowait()
            except queue.Empty:
                return

            clean_text = line.text.strip()
            source_time = line.end_time or line.timestamp
            # Suppress immediate snapshot duplicates while allowing another
            # speaker (or a later genuine repetition) to say the same words.
            if (
                clean_text == self._last_line_text
                and line.speaker_id == self._last_line_speaker
                and self._last_line_source_time is not None
                and abs(source_time - self._last_line_source_time) < 0.05
            ):
                logger.debug("Duplicate committed line ignored: %r", line.text)
                continue

            if (
                self._lines
                and self._current_speaker_id is not None
                and line.speaker_id is not None
                and line.speaker_id != self._current_speaker_id
            ):
                logger.info("SPEAKER_CHANGE from=%s to=%s", self._current_speaker_id, line.speaker_id)
                self._emit()

            self._last_line_text = clean_text
            self._last_line_speaker = line.speaker_id
            self._last_line_source_time = source_time

            if not self._lines:
                self._utterance_start = line.start_time or line.timestamp
                self._utterance_started_at = time.time()
                self._current_speaker_id = line.speaker_id
                self._current_speaker_label = line.speaker_label

            self._lines.append(line)
            self._last_line_time = time.time()
            logger.info(
                "BEHAVIOUR_TRACE aggregator_received transcript=%r line_count=%d timestamp=%.3f",
                line.text,
                len(self._lines),
                line.timestamp,
            )
            logger.debug("Accumulated committed line: %r", line.text)

    def _emit(self) -> None:
        """Build an ``Utterance`` from accumulated lines and enqueue it."""
        if not self._lines:
            return

        trace = self._lines[-1].latency_trace
        utterance = Utterance(
            lines=list(self._lines),
            start_time=self._utterance_start or self._lines[0].timestamp,
            end_time=self._lines[-1].end_time or self._lines[-1].timestamp,
            latency_trace=trace,
            speaker_id=self._current_speaker_id,
            speaker_label=self._current_speaker_label,
        )
        self._lines.clear()
        self._utterance_start = None
        self._utterance_started_at = None
        self._last_line_time = None
        self._last_line_text = ""
        self._last_line_speaker = None
        self._last_line_source_time = None
        self._current_speaker_id = None
        self._current_speaker_label = None

        try:
            self._utterance_queue.put_nowait(utterance)
            self._emitted_count += 1
            logger.info(
                "Utterance emitted — text=%r duration=%.2fs latency=%s",
                utterance.full_text[:60],
                utterance.duration(),
                trace.durations_ms() if trace else {},
            )
            logger.info(
                "BEHAVIOUR_TRACE aggregator_emitted transcript=%r start=%.3f end=%.3f utterance_q=%d",
                utterance.full_text,
                utterance.start_time,
                utterance.end_time,
                self._utterance_queue.qsize(),
            )
        except queue.Full:
            logger.warning("utterance_queue full — utterance dropped")

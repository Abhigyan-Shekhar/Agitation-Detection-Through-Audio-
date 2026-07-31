"""Tests for the utterance aggregator."""
from __future__ import annotations

import queue
import time
import pytest

from event_models import CommittedLine, Utterance
from utterance_aggregator import UtteranceAggregator


def _send_line(committed_q: queue.Queue, text: str, delay: float = 0.0) -> None:
    if delay:
        time.sleep(delay)
    committed_q.put(CommittedLine(text=text, timestamp=time.time()))


class TestUtteranceAggregator:
    def setup_method(self):
        self.committed_q: queue.Queue[CommittedLine] = queue.Queue()
        self.utterance_q: queue.Queue[Utterance] = queue.Queue()

    def _make_aggregator(self, silence_sec: float = 0.3, max_sec: float = 5.0) -> UtteranceAggregator:
        return UtteranceAggregator(
            committed_queue=self.committed_q,
            utterance_queue=self.utterance_q,
            silence_sec=silence_sec,
            max_utterance_sec=max_sec,
        )

    def test_single_line_emitted_after_silence(self):
        agg = self._make_aggregator(silence_sec=0.3)
        agg.start()
        try:
            _send_line(self.committed_q, "Hello world.")
            time.sleep(0.6)   # wait for silence timeout
            assert not self.utterance_q.empty()
            utterance = self.utterance_q.get_nowait()
            assert "Hello world" in utterance.full_text
        finally:
            agg.stop()

    def test_multiple_lines_assembled(self):
        agg = self._make_aggregator(silence_sec=0.3)
        agg.start()
        try:
            for text in ["Why can't I", "go home?"]:
                _send_line(self.committed_q, text)
                time.sleep(0.05)
            time.sleep(0.6)
            assert not self.utterance_q.empty()
            utterance = self.utterance_q.get_nowait()
            assert len(utterance.lines) == 2
        finally:
            agg.stop()

    def test_duplicate_lines_deduplicated(self):
        agg = self._make_aggregator(silence_sec=0.3)
        agg.start()
        try:
            for _ in range(3):
                _send_line(self.committed_q, "Same text.")
            time.sleep(0.6)
            assert not self.utterance_q.empty()
            utterance = self.utterance_q.get_nowait()
            assert len(utterance.lines) == 1
        finally:
            agg.stop()

    def test_flush_emits_pending(self):
        agg = self._make_aggregator(silence_sec=10.0)  # very long silence
        agg.start()
        try:
            _send_line(self.committed_q, "Flush me.")
            time.sleep(0.1)   # let aggregator consume the line
            agg.flush()
            time.sleep(0.1)
            assert not self.utterance_q.empty()
        finally:
            agg.stop()

    def test_stop_drains_committed_queue_before_flush(self):
        agg = self._make_aggregator(silence_sec=10.0)
        _send_line(self.committed_q, "Final line.")

        agg.stop()

        assert not self.utterance_q.empty()
        assert self.utterance_q.get_nowait().full_text == "Final line."

    def test_utterance_full_text_property(self):
        agg = self._make_aggregator(silence_sec=0.3)
        agg.start()
        try:
            for text in ["I want", "to go home."]:
                _send_line(self.committed_q, text)
                time.sleep(0.05)
            time.sleep(0.6)
            utterance = self.utterance_q.get_nowait()
            assert "I want" in utterance.full_text
            assert "go home" in utterance.full_text
        finally:
            agg.stop()

    def test_empty_queue_produces_no_utterance(self):
        agg = self._make_aggregator(silence_sec=0.3)
        agg.start()
        try:
            time.sleep(0.5)
            assert self.utterance_q.empty()
        finally:
            agg.stop()

    def test_max_duration_fires(self):
        agg = self._make_aggregator(silence_sec=5.0, max_sec=0.3)
        agg.start()
        try:
            # Send lines fast enough to keep silence timer alive
            for _ in range(3):
                _send_line(self.committed_q, f"Line {_}")
                time.sleep(0.05)
            time.sleep(0.5)   # max_sec=0.3 should have fired
            assert not self.utterance_q.empty()
        finally:
            agg.stop()

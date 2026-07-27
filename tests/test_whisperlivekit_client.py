"""Tests for WhisperLiveKit transcript routing."""
from __future__ import annotations

import queue

from event_models import CommittedLine
from whisperlivekit_client import WhisperLiveKitClient


def test_committed_dispatch_fans_out_to_all_queues() -> None:
    analysis_q: queue.Queue[CommittedLine] = queue.Queue()
    display_q: queue.Queue[CommittedLine] = queue.Queue()

    client = WhisperLiveKitClient.__new__(WhisperLiveKitClient)
    client._committed_queues = [analysis_q, display_q]
    client._committed_count = 0
    client._emitted_line_keys = set()

    client._dispatch({"type": "committed", "text": "Hello dashboard."})

    assert analysis_q.get_nowait().text == "Hello dashboard."
    assert display_q.get_nowait().text == "Hello dashboard."
    assert client._committed_count == 1


def test_wlk_buffer_transcription_updates_partial_caption() -> None:
    partial_q: queue.Queue[str] = queue.Queue()
    committed_q: queue.Queue[CommittedLine] = queue.Queue()

    client = WhisperLiveKitClient.__new__(WhisperLiveKitClient)
    client._partial_queue = partial_q
    client._committed_queues = [committed_q]
    client._partial_count = 0
    client._committed_count = 0
    client._emitted_line_keys = set()

    client._dispatch({"status": "active_transcription", "buffer_transcription": "hello wor"})

    assert partial_q.get_nowait() == "hello wor"
    assert committed_q.empty()
    assert client._partial_count == 1


def test_wlk_lines_snapshot_emits_only_new_committed_lines() -> None:
    committed_q: queue.Queue[CommittedLine] = queue.Queue()

    client = WhisperLiveKitClient.__new__(WhisperLiveKitClient)
    client._partial_queue = queue.Queue()
    client._committed_queues = [committed_q]
    client._partial_count = 0
    client._committed_count = 0
    client._emitted_line_keys = set()

    msg = {
        "status": "active_transcription",
        "lines": [
            {
                "speaker": 0,
                "start": "0:00:00",
                "end": "0:00:01",
                "text": "Hello dashboard.",
            }
        ],
        "buffer_transcription": "",
    }

    client._dispatch(msg)
    client._dispatch(msg)

    assert committed_q.get_nowait().text == "Hello dashboard."
    assert committed_q.empty()
    assert client._committed_count == 1


def test_wlk_diff_new_lines_are_committed() -> None:
    committed_q: queue.Queue[CommittedLine] = queue.Queue()

    client = WhisperLiveKitClient.__new__(WhisperLiveKitClient)
    client._partial_queue = queue.Queue()
    client._committed_queues = [committed_q]
    client._partial_count = 0
    client._committed_count = 0
    client._emitted_line_keys = set()

    client._dispatch({
        "type": "diff",
        "new_lines": [
            {
                "speaker": 0,
                "start": "0:00:01",
                "end": "0:00:02",
                "text": "This is final.",
            }
        ],
    })

    assert committed_q.get_nowait().text == "This is final."
    assert client._committed_count == 1

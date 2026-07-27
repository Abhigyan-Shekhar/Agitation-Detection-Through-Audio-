"""Tests for WhisperLiveKit transcript routing."""
from __future__ import annotations

import queue

from event_models import CommittedLine
from whisperlivekit_client import WhisperLiveKitClient


def _client_for_dispatch(
    partial_q: queue.Queue[str] | None = None,
    committed_q: queue.Queue[CommittedLine] | None = None,
) -> WhisperLiveKitClient:
    client = WhisperLiveKitClient.__new__(WhisperLiveKitClient)
    client._partial_queue = partial_q or queue.Queue()
    client._committed_queues = [committed_q or queue.Queue()]
    client._partial_count = 0
    client._committed_count = 0
    client._messages_received = 0
    client._bytes_sent = 0
    client._last_message_type = ""
    client._last_message_text = ""
    client._last_partial_text = ""
    client._last_committed_text = ""
    client._line_snapshot_text = ""
    client._line_snapshot_updated_at = 0.0
    client._line_snapshot_has_silence = False
    client._last_error = None
    client._emitted_line_keys = set()
    return client


def test_committed_dispatch_fans_out_to_all_queues() -> None:
    analysis_q: queue.Queue[CommittedLine] = queue.Queue()
    display_q: queue.Queue[CommittedLine] = queue.Queue()

    client = _client_for_dispatch(committed_q=analysis_q)
    client._committed_queues = [analysis_q, display_q]

    client._dispatch({"type": "committed", "text": "Hello dashboard."})

    assert analysis_q.get_nowait().text == "Hello dashboard."
    assert display_q.get_nowait().text == "Hello dashboard."
    assert client._committed_count == 1


def test_wlk_buffer_transcription_updates_partial_caption() -> None:
    partial_q: queue.Queue[str] = queue.Queue()
    committed_q: queue.Queue[CommittedLine] = queue.Queue()

    client = _client_for_dispatch(partial_q, committed_q)

    client._dispatch({"status": "active_transcription", "buffer_transcription": "hello wor"})

    assert partial_q.get_nowait() == "hello wor"
    assert committed_q.empty()
    assert client._partial_count == 1


def test_partial_caption_queue_keeps_latest_value_when_full() -> None:
    partial_q: queue.Queue[str] = queue.Queue(maxsize=1)
    partial_q.put_nowait("")

    client = _client_for_dispatch(partial_q=partial_q)

    client._put_partial("new live speech")

    assert partial_q.get_nowait() == "new live speech"
    assert client._partial_count == 1


def test_wlk_lines_snapshot_updates_partial_without_committing() -> None:
    partial_q: queue.Queue[str] = queue.Queue()
    committed_q: queue.Queue[CommittedLine] = queue.Queue()

    client = _client_for_dispatch(partial_q, committed_q)

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

    assert committed_q.empty()
    assert partial_q.get_nowait() == "Hello dashboard."
    assert partial_q.empty()
    assert client._partial_count == 1


def test_wlk_incremental_line_snapshots_commit_only_final_text() -> None:
    partial_q: queue.Queue[str] = queue.Queue()
    committed_q: queue.Queue[CommittedLine] = queue.Queue()
    client = _client_for_dispatch(partial_q, committed_q)

    for text in [
        "Hello, hello",
        "Hello, hello, hello",
        "Hello, hello, hello, please help me now",
    ]:
        client._dispatch({
            "status": "active_transcription",
            "lines": [
                {
                    "speaker": 1,
                    "start": "0:00:00",
                    "end": "0:00:02",
                    "text": text,
                }
            ],
            "buffer_transcription": "",
        })

    client._dispatch({
        "status": "no_audio_detected",
        "buffer_transcription": "",
    })

    partials = []
    while not partial_q.empty():
        partials.append(partial_q.get_nowait())

    assert partials[-1] == "Hello, hello, hello, please help me now"
    assert committed_q.get_nowait().text == "Hello, hello, hello, please help me now"
    assert committed_q.empty()
    assert client._committed_count == 1


def test_silent_line_snapshot_commits_after_stability_window(monkeypatch) -> None:
    committed_q: queue.Queue[CommittedLine] = queue.Queue()
    client = _client_for_dispatch(committed_q=committed_q)
    current_time = 100.0

    def fake_monotonic() -> float:
        return current_time

    monkeypatch.setattr("whisperlivekit_client.time.monotonic", fake_monotonic)

    msg = {
        "status": "active_transcription",
        "lines": [
            {
                "speaker": 1,
                "start": "0:00:00",
                "end": "0:00:02",
                "text": "please help me now",
            },
            {
                "speaker": -2,
                "start": "0:00:03",
                "end": "0:00:03",
                "text": "",
            },
        ],
        "buffer_transcription": "",
    }

    client._dispatch(msg)
    assert committed_q.empty()

    current_time = 100.4
    client._dispatch(msg)
    assert committed_q.empty()

    current_time = 100.9
    client._dispatch(msg)
    assert committed_q.get_nowait().text == "please help me now"
    assert committed_q.empty()
    assert client._committed_count == 1

    current_time = 101.8
    client._dispatch(msg)
    assert committed_q.empty()
    assert client._committed_count == 1


def test_wlk_diff_new_lines_are_committed() -> None:
    committed_q: queue.Queue[CommittedLine] = queue.Queue()

    client = _client_for_dispatch(committed_q=committed_q)

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


def test_empty_wlk_buffer_does_not_clear_live_partial_caption() -> None:
    partial_q: queue.Queue[str] = queue.Queue()
    committed_q: queue.Queue[CommittedLine] = queue.Queue()
    client = _client_for_dispatch(partial_q, committed_q)

    client._dispatch({
        "status": "active_transcription",
        "buffer_transcription": "please help me",
    })
    client._dispatch({
        "status": "active_transcription",
        "buffer_transcription": "",
        "lines": [],
    })

    assert partial_q.get_nowait() == "please help me"
    assert partial_q.empty()
    assert client.stats["last_message_text"] == "please help me"


def test_empty_wlk_buffer_commits_last_partial_when_no_line_arrives() -> None:
    partial_q: queue.Queue[str] = queue.Queue()
    committed_q: queue.Queue[CommittedLine] = queue.Queue()
    client = _client_for_dispatch(partial_q, committed_q)

    client._dispatch({
        "status": "active_transcription",
        "buffer_transcription": "please help me now",
    })
    client._dispatch({
        "status": "no_audio_detected",
        "buffer_transcription": "",
    })

    assert committed_q.get_nowait().text == "please help me now"
    assert committed_q.empty()
    assert client._committed_count == 1


def test_partial_fallback_does_not_duplicate_existing_committed_line() -> None:
    committed_q: queue.Queue[CommittedLine] = queue.Queue()
    client = _client_for_dispatch(committed_q=committed_q)

    client._dispatch({
        "status": "active_transcription",
        "buffer_transcription": "hello dashboard",
    })
    client._dispatch({
        "status": "active_transcription",
        "lines": [
            {
                "speaker": 0,
                "start": "0:00:00",
                "end": "0:00:01",
                "text": "hello dashboard",
            }
        ],
        "buffer_transcription": "",
    })
    client._dispatch({
        "status": "no_audio_detected",
        "buffer_transcription": "",
    })

    assert committed_q.get_nowait().text == "hello dashboard"
    assert committed_q.empty()
    assert client._committed_count == 1

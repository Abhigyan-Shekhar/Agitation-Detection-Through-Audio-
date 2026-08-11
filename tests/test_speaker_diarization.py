from __future__ import annotations

import asyncio
import json
import queue

import numpy as np
import pytest

import config
from baseline_manager import BaselineManager
from audio_pipeline import TimestampedFrame
from behaviour_classifier import BehaviourClassifier
from dashboard_manager import DashboardManager, DashboardStartupError
from event_models import CommittedLine, FusedResult, LinguisticFeatures, Utterance
from linguistic_features import LinguisticAnalyzer
from queue_fanout import publish_latest
from score_fusion import ScoreFusion
from speaker_utils import SpeakerRegistry, parse_wlk_timestamp, wlk_relative_to_wallclock
from utterance_aggregator import UtteranceAggregator
from whisperlivekit_client import WhisperLiveKitClient


def _client(*targets: queue.Queue[CommittedLine]) -> WhisperLiveKitClient:
    client = WhisperLiveKitClient(
        wlk_queue=queue.Queue(),
        partial_queue=queue.Queue(maxsize=5),
        committed_queue=list(targets) or [queue.Queue()],
        speaker_registry=SpeakerRegistry(),
        diarization_enabled=True,
    )
    client._stream_start_wallclock = 1_000.0
    return client


def _utterance(text: str, speaker_id: int | None, end: float) -> Utterance:
    label = None if speaker_id is None else f"Speaker {speaker_id}"
    line = CommittedLine(
        text=text,
        timestamp=end,
        speaker_id=speaker_id,
        speaker_label=label,
    )
    return Utterance(
        lines=[line],
        start_time=end - 1,
        end_time=end,
        speaker_id=speaker_id,
        speaker_label=label,
    )


def test_committed_line_constructor_remains_backwards_compatible() -> None:
    line = CommittedLine(text="hello", timestamp=123.0)
    assert line.speaker_id is None
    assert line.start_time is None


def test_wlk_full_snapshot_preserves_speakers_and_segment_times() -> None:
    output: queue.Queue[CommittedLine] = queue.Queue()
    client = _client(output)
    message = {
        "lines": [
            {"speaker": 1, "text": "Where am I?", "start": "0:00:01", "end": "0:00:03"},
            {"speaker": 2, "text": "You're at home.", "start": "0:00:03", "end": "0:00:05"},
        ],
        "buffer_transcription": "",
        "remaining_time_diarization": 0.25,
    }

    client._dispatch(message)

    first = output.get_nowait()
    second = output.get_nowait()
    assert (first.speaker_id, first.speaker_label, first.start_time, first.end_time) == (
        1,
        "Speaker 1",
        1001.0,
        1003.0,
    )
    assert (second.speaker_id, second.speaker_label, second.timestamp) == (
        2,
        "Speaker 2",
        1005.0,
    )
    assert client.stats["remaining_time_diarization"] == pytest.approx(0.25)
    assert client.stats["speakers_seen"] == [1, 2]


def test_wlk_repeated_snapshots_do_not_duplicate_and_silence_is_ignored() -> None:
    output: queue.Queue[CommittedLine] = queue.Queue()
    client = _client(output)
    message = {
        "type": "snapshot",
        "lines": [
            {"speaker": -2, "text": None, "start": "0:00:00", "end": "0:00:01"},
            {"speaker": 1, "text": "Hello", "start": "0:00:01", "end": "0:00:02"},
        ],
    }
    client._dispatch(message)
    client._dispatch(message)

    assert output.qsize() == 1
    assert output.get_nowait().text == "Hello"


def test_wlk_diff_lines_are_committed_and_partial_has_no_fabricated_speaker() -> None:
    output: queue.Queue[CommittedLine] = queue.Queue()
    client = _client(output)
    client._dispatch({
        "type": "diff",
        "new_lines": [
            {"speaker": 3, "text": "Committed", "start": "0:00:02", "end": "0:00:03"}
        ],
        "buffer_transcription": "still changing",
    })
    assert output.get_nowait().speaker_id == 3
    assert client._partial_queue.get_nowait() == "still changing"


def test_disabled_diarization_preserves_single_speaker_behavior() -> None:
    output: queue.Queue[CommittedLine] = queue.Queue()
    client = WhisperLiveKitClient(
        wlk_queue=queue.Queue(),
        partial_queue=queue.Queue(),
        committed_queue=output,
        diarization_enabled=False,
    )
    client._stream_start_wallclock = 1000.0
    client._dispatch({
        "lines": [
            {"speaker": 1, "text": "ordinary transcript", "start": "0:00:01", "end": "0:00:02"}
        ]
    })
    line = output.get_nowait()
    assert line.speaker_id is None
    assert line.speaker_label is None
    assert client.stats["speakers_seen"] == []


def test_wlk_stop_flushes_pcm_before_end_of_audio_signal() -> None:
    client = _client(queue.Queue())
    client._wlk_queue.put(TimestampedFrame(np.array([0.25, -0.25], dtype=np.float32), 1000.0))
    client._stop_event.set()

    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        async def send(self, payload: bytes) -> None:
            self.sent.append(payload)
            if payload == b"":
                client._ready_to_stop_event.set()

    websocket = FakeWebSocket()
    asyncio.run(client._send_loop(websocket))
    assert websocket.sent[-1] == b""
    assert len(websocket.sent[0]) == 4


def test_wlk_stop_still_receives_final_committed_lines() -> None:
    output: queue.Queue[CommittedLine] = queue.Queue()
    client = _client(output)
    client._stop_event.set()
    messages = [
        json.dumps({
            "lines": [
                {"speaker": 1, "text": "final words", "start": "0:00:01", "end": "0:00:02"}
            ]
        }),
        json.dumps({"type": "ready_to_stop"}),
    ]

    class FakeWebSocket:
        def __aiter__(self):
            self._items = iter(messages)
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    asyncio.run(client._recv_loop(FakeWebSocket()))
    assert output.get_nowait().text == "final words"
    assert client._ready_to_stop_event.is_set()


def test_timestamp_parser_and_wallclock_alignment() -> None:
    assert parse_wlk_timestamp("1:02:03.5") == pytest.approx(3723.5)
    assert parse_wlk_timestamp("malformed") is None
    assert wlk_relative_to_wallclock(1_700_000_000.0, "0:00:03.25") == pytest.approx(
        1_700_000_003.25
    )


def test_aggregator_splits_on_speaker_change() -> None:
    committed: queue.Queue[CommittedLine] = queue.Queue()
    utterances: queue.Queue[Utterance] = queue.Queue()
    aggregator = UtteranceAggregator(committed, utterances, silence_sec=10)
    for speaker, text, start, end in [
        (1, "I want", 1.0, 2.0),
        (1, "to leave.", 2.0, 3.0),
        (2, "Please sit down.", 3.0, 4.0),
        (1, "No.", 4.0, 5.0),
    ]:
        committed.put(CommittedLine(
            text=text,
            timestamp=end,
            speaker_id=speaker,
            speaker_label=f"Speaker {speaker}",
            start_time=start,
            end_time=end,
        ))
    aggregator._drain_committed_queue()
    aggregator.flush()

    emitted = [utterances.get_nowait() for _ in range(3)]
    assert [item.speaker_id for item in emitted] == [1, 2, 1]
    assert [item.full_text for item in emitted] == [
        "I want to leave.",
        "Please sit down.",
        "No.",
    ]
    assert (emitted[0].start_time, emitted[0].end_time) == (1.0, 3.0)


def test_identical_text_from_different_speakers_is_not_deduplicated() -> None:
    committed: queue.Queue[CommittedLine] = queue.Queue()
    utterances: queue.Queue[Utterance] = queue.Queue()
    aggregator = UtteranceAggregator(committed, utterances)
    committed.put(CommittedLine("Same", 1.0, speaker_id=1, speaker_label="Speaker 1"))
    committed.put(CommittedLine("Same", 2.0, speaker_id=2, speaker_label="Speaker 2"))
    aggregator._drain_committed_queue()
    aggregator.flush()
    assert [utterances.get_nowait().speaker_id for _ in range(2)] == [1, 2]


def test_linguistic_repetition_history_is_partitioned_by_speaker() -> None:
    analyzer = LinguisticAnalyzer()
    first = analyzer.analyze(_utterance("Where am I?", 1, 10.0))
    other = analyzer.analyze(_utterance("Where am I?", 2, 11.0))
    repeated = analyzer.analyze(_utterance("Where am I?", 1, 12.0))
    assert first.question_repetition_score == 0.0
    assert other.question_repetition_score == 0.0
    assert repeated.question_repetition_score > 0.9


def test_behaviour_event_preserves_originating_speaker() -> None:
    utterance = _utterance("Help me now!", 1, 20.0)
    result = FusedResult(
        smoothed_score=0.8,
        severity="High",
        reliability=0.8,
        speaker_id=1,
        speaker_label="Speaker 1",
        utterance=utterance,
        linguistic_features=LinguisticFeatures(urgency_score=0.9),
    )
    classified = BehaviourClassifier().classify(result)
    assert classified.behaviour_events
    assert all(event.speaker_id == 1 for event in classified.behaviour_events)
    assert all(event.person == "Speaker 1" for event in classified.behaviour_events)


def test_score_smoothing_does_not_leak_between_speakers() -> None:
    fusion = ScoreFusion(BaselineManager())
    agitated = LinguisticFeatures(
        repetition_score=1.0,
        question_repetition_score=1.0,
        negative_sentiment=1.0,
        urgency_score=1.0,
        threat_score=1.0,
        profanity_score=1.0,
        sexual_advance_score=1.0,
        strange_noise_score=1.0,
    )
    first = fusion.fuse(_utterance("Help", 1, 30.0), None, agitated)
    other = fusion.fuse(_utterance("Hello", 2, 31.0), None, LinguisticFeatures())
    assert first.smoothed_score > 0.0
    assert other.smoothed_score == 0.0


def test_committed_queue_fanout_is_non_destructive() -> None:
    analysis: queue.Queue[CommittedLine] = queue.Queue()
    display: queue.Queue[CommittedLine] = queue.Queue()
    line = CommittedLine("hello", 1.0, speaker_id=1)
    assert publish_latest([analysis, display], line, logger=__import__("logging").getLogger(__name__), label="test") == 2
    assert display.get_nowait() is line
    assert analysis.get_nowait() is line


def test_wlk_command_enables_sortformer_only_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = DashboardManager(queue.Queue(), queue.Queue(), queue.Queue(), BaselineManager())
    monkeypatch.setattr(config, "ENABLE_SPEAKER_DIARIZATION", True)
    monkeypatch.setattr(config, "DIARIZATION_BACKEND", "sortformer")
    command = manager.wlk_command
    assert "--pcm-input" in command
    assert command[command.index("--diarization-backend") + 1] == "sortformer"
    assert "--diarization" in command
    assert "--max-speakers" not in command

    monkeypatch.setattr(config, "ENABLE_SPEAKER_DIARIZATION", False)
    assert "--diarization" not in manager.wlk_command


def test_unsupported_local_sortformer_fails_with_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DashboardManager(queue.Queue(), queue.Queue(), queue.Queue(), BaselineManager())
    monkeypatch.setattr(config, "WLK_AUTO_LAUNCH", True)
    monkeypatch.setattr(config, "ENABLE_SPEAKER_DIARIZATION", True)
    monkeypatch.setattr(config, "DIARIZATION_BACKEND", "sortformer")
    monkeypatch.setattr("dashboard_manager.platform.system", lambda: "Darwin")
    with pytest.raises(DashboardStartupError, match="Linux with Python 3.11-3.12"):
        manager._start_wlk()

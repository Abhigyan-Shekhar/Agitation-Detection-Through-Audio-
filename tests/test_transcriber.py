import queue
import time
from types import SimpleNamespace

import numpy as np

from audio_pipeline import TimestampedFrame
from event_models import CommittedLine
from transcriber import DirectWhisperTranscriber, TranscriptionWorker


class FakeModel:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        return [SimpleNamespace(text="hello world", start=0.0, end=0.5, avg_logprob=-0.1)], SimpleNamespace(language="en")


def frame(samples, idx=1):
    return TimestampedFrame(
        data=np.ones(samples, dtype=np.float32) * 0.1,
        timestamp=time.time(),
        capture_monotonic=time.monotonic(),
        queued_monotonic=time.monotonic(),
        frame_index=idx,
    )


def test_direct_transcriber_reuses_injected_model_and_returns_confidence():
    model = FakeModel()
    transcriber = DirectWhisperTranscriber(model="small", model_size="small") if False else DirectWhisperTranscriber(model_size="small", model=model)

    text, segments, confidence = transcriber.transcribe(np.ones(1600, dtype=np.float32))

    assert model.calls == 1
    assert text == "hello world"
    assert segments[0].start == 0.0
    assert confidence is not None


def test_worker_rolls_buffer_and_emits_transcript_without_blocking():
    audio_q = queue.Queue()
    partial_q = queue.Queue(maxsize=2)
    committed_q = queue.Queue(maxsize=2)
    transcriber = DirectWhisperTranscriber(model_size="small", model=FakeModel())
    worker = TranscriptionWorker(
        audio_queue=audio_q,
        partial_queue=partial_q,
        committed_queue=committed_q,
        transcriber=transcriber,
        window_seconds=0.1,
        interval_seconds=0.01,
        sample_rate=1000,
    )

    for i in range(5):
        audio_q.put(frame(40, i + 1))

    worker._drain_audio_queue()
    assert worker._sample_count <= 100

    worker._transcribe_buffer()

    assert partial_q.get_nowait() == "hello world"
    committed = committed_q.get_nowait()
    assert isinstance(committed, CommittedLine)
    assert committed.text == "hello world"
    assert worker.latest_result is not None
    assert worker.latest_result.buffer_duration <= 0.1


def test_worker_logs_and_continues_when_transcription_fails(caplog):
    class BrokenTranscriber:
        def transcribe(self, audio):
            raise RuntimeError("boom")

    worker = TranscriptionWorker(
        audio_queue=queue.Queue(),
        partial_queue=queue.Queue(),
        committed_queue=queue.Queue(),
        transcriber=BrokenTranscriber(),
        window_seconds=1,
        interval_seconds=1,
        sample_rate=1000,
    )
    worker._frames.append(frame(10))
    worker._sample_count = 10

    worker._transcribe_buffer()

    assert "Transcription failed" in caplog.text

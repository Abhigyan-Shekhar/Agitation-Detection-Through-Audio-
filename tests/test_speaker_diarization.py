from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np

from audio_pipeline import TimestampedFrame
from baseline_manager import BaselineManager
from behaviour_classifier import BehaviourClassifier
from event_models import CommittedLine, LinguisticFeatures, Utterance
from linguistic_features import LinguisticAnalyzer
from speaker_diarization import OnlineSpeakerDiarizer
from score_fusion import ScoreFusion
from transcriber import DirectWhisperTranscriber, TranscriptionWorker
from utterance_aggregator import UtteranceAggregator


class SequenceEmbeddingBackend:
    def __init__(self, vectors):
        self.vectors = iter(vectors)

    def embed(self, audio, sample_rate):
        return np.asarray(next(self.vectors), dtype=np.float32)


def test_online_clustering_keeps_stable_session_speaker_ids():
    backend = SequenceEmbeddingBackend([[1, 0], [0.98, 0.02], [0, 1], [0.01, 0.99]])
    diarizer = OnlineSpeakerDiarizer(backend=backend, min_segment_seconds=0, similarity_threshold=0.8)
    audio = np.ones(100, dtype=np.float32)

    ids = [diarizer.identify(audio, 16_000)[0] for _ in range(4)]

    assert ids == [1, 1, 2, 2]
    assert diarizer.speakers_seen == 2


def test_short_audio_is_not_given_a_fabricated_speaker():
    diarizer = OnlineSpeakerDiarizer(
        backend=SequenceEmbeddingBackend([[1, 0]]), min_segment_seconds=1.0
    )
    assert diarizer.identify(np.ones(100), 16_000) == (None, None)


class TwoSegmentModel:
    def transcribe(self, audio, **kwargs):
        return [
            SimpleNamespace(text="I want to leave.", start=0.0, end=1.0, avg_logprob=-0.1),
            SimpleNamespace(text="Please sit down.", start=1.0, end=2.0, avg_logprob=-0.1),
        ], SimpleNamespace(language="en")


def test_local_transcriber_emits_speaker_tagged_timed_segments_without_wlk():
    audio_q = queue.Queue()
    committed_q = queue.Queue()
    diarizer = OnlineSpeakerDiarizer(
        backend=SequenceEmbeddingBackend([[1, 0], [0, 1]]),
        min_segment_seconds=0,
        similarity_threshold=0.8,
    )
    worker = TranscriptionWorker(
        audio_queue=audio_q,
        partial_queue=queue.Queue(),
        committed_queue=committed_q,
        transcriber=DirectWhisperTranscriber(model_size="small", model=TwoSegmentModel()),
        diarizer=diarizer,
        enable_diarization=True,
        sample_rate=1000,
        window_seconds=3,
    )
    audio_q.put(TimestampedFrame(np.ones(2000, dtype=np.float32), 1000.0))

    worker._drain_audio_queue()
    worker._transcribe_buffer()
    lines = [committed_q.get_nowait(), committed_q.get_nowait()]

    assert [(line.speaker_id, line.speaker_label) for line in lines] == [
        (1, "Speaker 1"), (2, "Speaker 2")
    ]
    assert (lines[0].start_time, lines[0].end_time) == (1000.0, 1001.0)
    assert (lines[1].start_time, lines[1].end_time) == (1001.0, 1002.0)

    worker._transcribe_buffer()
    assert committed_q.empty(), "rolling snapshots must not duplicate committed segments"


def test_speaker_change_finalizes_utterance_and_same_text_is_not_cross_speaker_duplicate():
    committed_q = queue.Queue()
    utterance_q = queue.Queue()
    agg = UtteranceAggregator(committed_q, utterance_q, silence_sec=10)
    committed_q.put(CommittedLine("Where am I?", 1.0, speaker_id=1, speaker_label="Speaker 1"))
    committed_q.put(CommittedLine("Where am I?", 2.0, speaker_id=2, speaker_label="Speaker 2"))
    committed_q.put(CommittedLine("No.", 3.0, speaker_id=1, speaker_label="Speaker 1"))

    agg.stop()
    utterances = [utterance_q.get_nowait() for _ in range(3)]

    assert [item.speaker_id for item in utterances] == [1, 2, 1]
    assert [item.full_text for item in utterances] == ["Where am I?", "Where am I?", "No."]


def _utterance(text: str, speaker_id: int, timestamp: float) -> Utterance:
    line = CommittedLine(text, timestamp, speaker_id=speaker_id)
    return Utterance([line], timestamp - 1, timestamp, speaker_id=speaker_id)


def test_repetition_history_is_scoped_per_speaker():
    analyzer = LinguisticAnalyzer()
    analyzer.analyze(_utterance("Where am I?", 1, 10.0))

    other = analyzer.analyze(_utterance("Where am I?", 2, 11.0))
    same = analyzer.analyze(_utterance("Where am I?", 1, 12.0))

    assert other.question_repetition_score == 0.0
    assert same.question_repetition_score > 0.5


def test_disabled_diarization_preserves_unlabelled_single_speaker_mode():
    worker = TranscriptionWorker(
        audio_queue=queue.Queue(),
        partial_queue=queue.Queue(),
        committed_queue=queue.Queue(),
        transcriber=DirectWhisperTranscriber(model_size="small", model=TwoSegmentModel()),
        enable_diarization=False,
        sample_rate=1000,
    )
    assert worker._identify_speaker(np.ones(1000, dtype=np.float32)) == (None, None)
    assert worker.diarization_active is False
    assert worker.diarization_error is None


def test_diarization_failure_is_exposed_for_dashboard_diagnostics():
    class BrokenDiarizer:
        speakers_seen = 0

        def reset(self):
            pass

        def identify(self, audio, sample_rate):
            raise RuntimeError("speaker model unavailable")

    worker = TranscriptionWorker(
        audio_queue=queue.Queue(),
        partial_queue=queue.Queue(),
        committed_queue=queue.Queue(),
        transcriber=DirectWhisperTranscriber(model_size="small", model=TwoSegmentModel()),
        diarizer=BrokenDiarizer(),
        enable_diarization=True,
        sample_rate=1000,
    )

    assert worker._identify_speaker(np.ones(1000, dtype=np.float32)) == (None, None)
    assert worker.diarization_active is False
    assert worker.diarization_error == "RuntimeError: speaker model unavailable"


def test_speaker_identity_reaches_fused_result_and_behaviour_event():
    utterance = _utterance("I will hit you.", 1, 20.0)
    utterance.speaker_label = "Speaker 1"
    linguistic = LinguisticFeatures(complaint_score=1.0, negative_sentiment=1.0)

    result = ScoreFusion(BaselineManager()).fuse(utterance, None, linguistic)
    result = BehaviourClassifier().classify(result)

    assert result.speaker_id == 1
    assert result.speaker_label == "Speaker 1"
    assert result.behaviour_events
    assert all(event.speaker_id == 1 for event in result.behaviour_events)
    assert all(event.speaker_label == "Speaker 1" for event in result.behaviour_events)

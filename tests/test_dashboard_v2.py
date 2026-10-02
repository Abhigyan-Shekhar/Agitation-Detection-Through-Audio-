from __future__ import annotations

import sys
import types

from qwen_person3 import FinalBehaviourResult


class _FakeDataFrame(list):
    def __init__(self, rows):
        super().__init__(rows)
        self._rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    @property
    def empty(self):
        return not self._rows

    @property
    def iloc(self):
        rows = self._rows

        class _ILoc:
            def __getitem__(self, idx):
                return rows[idx]

        return _ILoc()


def _install_dashboard_import_stubs():
    pandas = types.ModuleType("pandas")
    pandas.DataFrame = _FakeDataFrame
    numpy = types.ModuleType("numpy")
    streamlit = types.ModuleType("streamlit")
    batch = types.ModuleType("batch_transcription")
    batch.SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav"})
    batch.inspect_upload = lambda *_args, **_kwargs: None
    batch.iter_transcription_chunks = lambda *_args, **_kwargs: iter(())
    batch.preprocess_upload = lambda *_args, **_kwargs: None
    batch.transcribe_upload = lambda *_args, **_kwargs: None
    person2 = types.ModuleType("person2_module")
    person2.analyze_person1_transcript = lambda *_args, **_kwargs: None
    person2.prepare_embedding_provider = lambda *_args, **_kwargs: None
    sys.modules.setdefault("pandas", pandas)
    sys.modules.setdefault("numpy", numpy)
    sys.modules.setdefault("streamlit", streamlit)
    sys.modules.setdefault("batch_transcription", batch)
    sys.modules.setdefault("person2_module", person2)


def test_dashboard_formatting_helpers():
    _install_dashboard_import_stubs()
    from dashboard_v2 import final_results_table, format_timestamp, result_timestamp, timeline_table, transcript_rows, upload_cache_key

    result = FinalBehaviourResult(
        behaviour="Repetitive Questioning",
        start=10.2,
        end=27.5,
        validated=True,
        severity="Moderate",
        confidence=0.94,
        evidence="Repeated question.",
        explanation="Supported.",
        initial_behaviour="Repeated questioning",
        initial_score=0.91,
        person2_evidence="Phrase repeated.",
        transcript="Where is my daughter?",
    )

    assert format_timestamp(10.2) == "00:10.2"
    assert result_timestamp(result) == "00:10.2 – 00:27.5"
    assert transcript_rows([{"start": 10.2, "end": 15.4, "text": "Where is my daughter?"}])[0]["Timestamp"] == "00:10.2 – 00:15.4"
    assert list(final_results_table([result]).columns) == ["Behaviour", "Timestamp", "Severity", "Confidence", "Validated"]
    assert timeline_table([result]).iloc[0]["start_sec"] == 10.2
    assert upload_cache_key(b"audio", "recording.wav") == upload_cache_key(b"audio", "recording.wav")
    assert upload_cache_key(b"audio", "recording.wav") != upload_cache_key(b"changed", "recording.wav")


def test_timeline_events_keep_exact_records_transcripts_and_audio_bounds():
    _install_dashboard_import_stubs()
    from dashboard_v2 import _event_hover_text, audio_segment_bounds, selected_timeline_event, timeline_events

    repeated = FinalBehaviourResult(
        behaviour="Repetitive Questioning", start=134.2, end=147.8, validated=True,
        severity="Moderate", confidence=0.87, evidence="Repeated questions.", explanation="Supported.",
        initial_behaviour="Repeated questioning", initial_score=0.91, person2_evidence="Repeated.",
        transcript="fallback", chunk_id="chunk-1", evidence_segment_ids=["seg-153", "seg-142", "seg-147"],
    )
    overlapping = FinalBehaviourResult(
        behaviour="Complaining", start=140.0, end=150.0, validated=True,
        severity="Mild", confidence=0.71, evidence="Complaint.", explanation="Supported.",
        initial_behaviour="Complaining", initial_score=0.71, person2_evidence="Complaint.",
        transcript="complaint", chunk_id="chunk-2", evidence_segment_ids=["seg-200"],
    )
    behaviours = [{
        "chunk_id": "chunk-1",
        "evidence_segments": [
            {"id": "seg-153", "start": 146.0, "end": 147.8, "text": "Where's my daughter?"},
            {"id": "seg-142", "start": 134.2, "end": 137.1, "text": "Where is my daughter?"},
            {"id": "seg-147", "start": 141.4, "end": 144.1, "text": "Where is my daughter?"},
        ],
    }, {
        "chunk_id": "chunk-2",
        "evidence_segments": [{"id": "seg-200", "start": 140.0, "end": 150.0, "text": "This is terrible."}],
    }]

    events = timeline_events([repeated, overlapping], behaviours)
    selected_repeat = selected_timeline_event(events, 0)
    selected_overlap = selected_timeline_event(events, 1)

    assert selected_repeat["result"] is repeated
    assert audio_segment_bounds(selected_repeat) == (134.2, 147.8)
    assert [segment["id"] for segment in selected_repeat["transcript_segments"]] == ["seg-142", "seg-147", "seg-153"]
    assert [segment["text"] for segment in selected_repeat["transcript_segments"]] == [
        "Where is my daughter?", "Where is my daughter?", "Where's my daughter?",
    ]
    hover_text = _event_hover_text(selected_repeat)
    assert "Confidence: 0.87" in hover_text
    assert "Status: Supported" in hover_text
    assert hover_text.index("seg-142") < hover_text.index("seg-147") < hover_text.index("seg-153")
    # The overlapping bar is an independent event identity and keeps its own bounds.
    assert selected_overlap["result"] is overlapping
    assert audio_segment_bounds(selected_overlap) == (140.0, 150.0)

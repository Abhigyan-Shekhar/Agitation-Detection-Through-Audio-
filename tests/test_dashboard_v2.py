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
    streamlit = types.ModuleType("streamlit")
    batch = types.ModuleType("batch_transcription")
    batch.SUPPORTED_AUDIO_EXTENSIONS = frozenset({".wav"})
    batch.preprocess_upload = lambda *_args, **_kwargs: None
    batch.transcribe_upload = lambda *_args, **_kwargs: None
    person2 = types.ModuleType("person2_module")
    person2.analyze_person1_transcript = lambda *_args, **_kwargs: None
    sys.modules.setdefault("pandas", pandas)
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

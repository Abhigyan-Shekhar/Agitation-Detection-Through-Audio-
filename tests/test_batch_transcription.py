from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from batch_transcription import (
    AudioValidationError,
    inspect_upload,
    iter_transcription_chunks,
    preprocess_upload,
    transcribe_upload,
    validate_upload,
)
from batch_transcription import _acoustic_only_events


def _wav_bytes(*, seconds: float = 0.25, sample_rate: int = 8_000, channels: int = 2) -> bytes:
    count = int(seconds * sample_rate)
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(count) / sample_rate) * 32767).astype("<i2")
    interleaved = np.repeat(tone[:, None], channels, axis=1).reshape(-1)
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(interleaved.tobytes())
    return output.getvalue()


@pytest.mark.parametrize("name", ["recording.txt", "recording", "recording.exe"])
def test_validate_upload_rejects_unsupported_extensions(name):
    with pytest.raises(AudioValidationError, match="Unsupported audio type"):
        validate_upload(name, b"not empty")


def test_validate_upload_rejects_empty_and_oversized_files():
    with pytest.raises(AudioValidationError, match="empty"):
        validate_upload("recording.wav", b"")
    with pytest.raises(AudioValidationError, match="upload limit"):
        validate_upload("recording.wav", b"12", max_bytes=1)


def test_preprocess_upload_decodes_stereo_and_resamples_to_whisper_format():
    processed = preprocess_upload(_wav_bytes(), "../recording.wav")

    assert processed.filename == "recording.wav"
    assert processed.sample_rate == 16_000
    assert processed.samples.dtype == np.float32
    assert processed.samples.ndim == 1
    assert processed.duration == pytest.approx(0.25, abs=0.01)
    assert np.isfinite(processed.samples).all()
    assert np.max(np.abs(processed.samples)) <= 1.0


def test_inspect_upload_reports_duration_without_full_preprocess():
    metadata = inspect_upload(_wav_bytes(seconds=1.25), "../recording.wav")

    assert metadata.filename == "recording.wav"
    assert metadata.duration == pytest.approx(1.25, abs=0.02)
    assert metadata.source_bytes > 0


def test_iter_transcription_chunks_covers_audio_with_overlap_and_final_partial():
    chunks = list(
        iter_transcription_chunks(
            _wav_bytes(seconds=2.5),
            "long.wav",
            chunk_seconds=1.0,
            overlap_seconds=0.2,
        )
    )

    assert [(round(c.primary_start, 1), round(c.primary_end, 1)) for c in chunks] == [(0.0, 1.0), (1.0, 2.0), (2.0, 2.5)]
    assert chunks[0].input_start == pytest.approx(0.0)
    assert chunks[1].input_start == pytest.approx(0.8, abs=0.01)
    assert chunks[2].input_start == pytest.approx(1.8, abs=0.01)
    assert sum(c.primary_end - c.primary_start for c in chunks) == pytest.approx(2.5, abs=0.02)


def test_preprocess_upload_rejects_fake_audio_with_allowed_extension():
    with pytest.raises(AudioValidationError, match="decoded"):
        preprocess_upload(b"this is not audio", "fake.mp3")


def test_transcribe_upload_returns_timestamped_person2_contract():
    class FakeTranscriber:
        model_size = "test-model"

        def transcribe(self, samples):
            assert samples.dtype == np.float32
            return (
                "Where is my daughter? Where is my daughter?",
                [
                    SimpleNamespace(text=" Where is my daughter? ", start=0.01, end=0.12, confidence=0.9),
                    SimpleNamespace(text="Where is my daughter?", start=0.13, end=99.0, confidence=1.2),
                ],
                0.9,
            )

    result = transcribe_upload(_wav_bytes(), "question.wav", transcriber=FakeTranscriber())

    assert result.sample_rate == 16_000
    assert result.model == "test-model"
    assert result.transcript_contract() == [
        {"start": 0.01, "end": 0.12, "text": "Where is my daughter?", "confidence": 0.9},
        {"start": 0.13, "end": 0.25, "text": "Where is my daughter?", "confidence": 1.0},
    ]


def test_transcribe_upload_offsets_chunk_timestamps_without_duplicating_overlap():
    class FakeTranscriber:
        model_size = "test-model"

        def __init__(self):
            self.calls = 0

        def transcribe(self, samples):
            self.calls += 1
            if self.calls == 1:
                return "first boundary", [SimpleNamespace(text="first", start=0.1, end=0.4), SimpleNamespace(text="boundary", start=0.9, end=1.1)], None
            if self.calls == 2:
                return "overlap second", [SimpleNamespace(text="overlap", start=0.05, end=0.15), SimpleNamespace(text="second", start=0.3, end=0.6)], None
            return "third", [SimpleNamespace(text="third", start=0.25, end=0.45)], None

    transcriber = FakeTranscriber()
    result = transcribe_upload(
        _wav_bytes(seconds=2.4),
        "long.wav",
        transcriber=transcriber,
        chunk_seconds=1.0,
        overlap_seconds=0.2,
    )

    assert transcriber.calls == 3
    assert [(segment.text, segment.start, segment.end) for segment in result.segments] == [
        ("first", 0.1, 0.4),
        ("boundary", 0.9, 1.1),
        ("second", 1.1, 1.4),
        ("third", 2.05, 2.25),
    ]


def test_acoustic_only_persistence_rejects_silence():
    assert _acoustic_only_events(np.zeros(16_000, dtype=np.float32), 16_000, 0.01) == []

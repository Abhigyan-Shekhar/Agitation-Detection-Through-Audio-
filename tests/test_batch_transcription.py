from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import wave

import numpy as np
import pytest

import batch_transcription
import config
from batch_transcription import (
    AudioValidationError,
    BatchTranscriberLoadError,
    _normalise_segments,
    clear_batch_transcriber_cache,
    get_batch_transcriber,
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


@pytest.fixture
def isolated_batch_transcriber_cache():
    clear_batch_transcriber_cache()
    yield
    clear_batch_transcriber_cache()


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


def test_batch_upload_defaults_use_small_cpu_model_and_two_minute_chunks():
    assert config.BATCH_WHISPER_CPU_MODEL == "small"
    assert config.BATCH_WHISPER_GPU_MODEL == "large-v3"
    assert config.BATCH_TRANSCRIPTION_CHUNK_SECONDS == 120
    assert config.BATCH_WHISPER_BEAM_SIZE == 5
    assert config.BATCH_WHISPER_WORD_TIMESTAMPS is True


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


def test_word_timestamps_split_oversized_whisper_segment_into_evidence_units():
    raw = [
        SimpleNamespace(
            text="I am fine. I won't take my medicine.",
            start=0.0,
            end=20.0,
            words=[
                SimpleNamespace(word="I", start=0.0, end=0.1, probability=0.9),
                SimpleNamespace(word="am", start=0.1, end=0.2, probability=0.9),
                SimpleNamespace(word="fine.", start=0.2, end=0.5, probability=0.9),
                SimpleNamespace(word="I", start=12.0, end=12.1, probability=0.8),
                SimpleNamespace(word="won't", start=12.1, end=12.4, probability=0.8),
                SimpleNamespace(word="take", start=12.4, end=12.7, probability=0.8),
                SimpleNamespace(word="my", start=12.7, end=12.9, probability=0.8),
                SimpleNamespace(word="medicine.", start=12.9, end=13.4, probability=0.8),
            ],
        )
    ]

    units = list(_normalise_segments(raw, 30.0))

    assert [(unit.text, unit.start, unit.end) for unit in units] == [
        ("I am fine.", 0.0, 0.5),
        ("I won't take my medicine.", 12.0, 13.4),
    ]


def test_acoustic_only_persistence_rejects_silence():
    assert _acoustic_only_events(np.zeros(16_000, dtype=np.float32), 16_000, 0.01) == []


def test_cached_batch_transcriber_reuses_identical_configuration_and_separates_models(monkeypatch, isolated_batch_transcriber_cache):
    created = []

    class FakeDirectWhisperTranscriber:
        @staticmethod
        def _cuda_available():
            return False

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            created.append(kwargs)

    monkeypatch.setattr(batch_transcription, "DirectWhisperTranscriber", FakeDirectWhisperTranscriber)
    first = get_batch_transcriber(model_size="small", device="cpu", compute_type="int8")
    second = get_batch_transcriber(model_size="small", device="cpu", compute_type="int8")
    different = get_batch_transcriber(model_size="medium", device="cpu", compute_type="int8")

    assert first is second
    assert different is not first
    assert len(created) == 2
    assert first.beam_size == 5
    assert first.word_timestamps is True


def test_batch_transcriber_load_failure_has_actionable_context(monkeypatch, isolated_batch_transcriber_cache):
    class FailingTranscriber:
        @staticmethod
        def _cuda_available():
            return False

        def __init__(self, **kwargs):
            raise MemoryError("not enough RAM")

    monkeypatch.setattr(batch_transcription, "DirectWhisperTranscriber", FailingTranscriber)
    with pytest.raises(BatchTranscriberLoadError, match="small.*CPU.*not enough RAM"):
        get_batch_transcriber(model_size="small", device="cpu")


def test_transcription_progress_starts_without_full_recording_rms_prepass(monkeypatch):
    class FakeTranscriber:
        model_size = "test-model"

        def transcribe(self, samples):
            return "hello", [SimpleNamespace(text="hello", start=0.0, end=0.1)], 0.9

    monkeypatch.setattr(
        batch_transcription,
        "_recording_baseline_rms_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full-file RMS pre-pass used")),
    )
    updates = []
    result = transcribe_upload(
        _wav_bytes(seconds=0.25), "progress.wav", transcriber=FakeTranscriber(),
        progress_callback=lambda *args: updates.append(args),
    )

    assert result.segments
    messages = [item[3] for item in updates]
    assert messages[0] == "Validating audio..."
    assert "Preparing transcription..." in messages
    assert any(message.startswith("Transcribing chunk 1/1") for message in messages)


def test_model_progress_reports_real_load_then_reuse(monkeypatch, isolated_batch_transcriber_cache):
    class FakeDirectWhisperTranscriber:
        @staticmethod
        def _cuda_available():
            return False

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(batch_transcription, "DirectWhisperTranscriber", FakeDirectWhisperTranscriber)
    first_updates = []
    second_updates = []
    get_batch_transcriber(model_size="small", device="cpu", progress_callback=lambda *args: first_updates.append(args))
    get_batch_transcriber(model_size="small", device="cpu", progress_callback=lambda *args: second_updates.append(args))

    assert first_updates[0][0] == "loading_model"
    assert "Loading Whisper small on CPU" in first_updates[0][3]
    assert "ready" in first_updates[-1][3]
    assert "Reusing Whisper small on CPU" in second_updates[-1][3]


def test_gpu_batch_default_selects_large_v3_without_changing_quality_settings(monkeypatch, isolated_batch_transcriber_cache):
    class FakeDirectWhisperTranscriber:
        @staticmethod
        def _cuda_available():
            return True

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(batch_transcription, "DirectWhisperTranscriber", FakeDirectWhisperTranscriber)
    monkeypatch.setattr(config, "USE_GPU_IF_AVAILABLE", True)
    engine = get_batch_transcriber()

    assert engine.model_size == "large-v3"
    assert engine.device == "cuda"
    assert engine.compute_type == "float16"
    assert engine.beam_size == 5
    assert engine.word_timestamps is True

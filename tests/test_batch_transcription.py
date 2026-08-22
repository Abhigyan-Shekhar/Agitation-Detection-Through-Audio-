from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from batch_transcription import (
    AudioValidationError,
    preprocess_upload,
    transcribe_upload,
    validate_upload,
)


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

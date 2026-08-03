from __future__ import annotations

import numpy as np

import acoustic_vocalization_detector as detector


class _FakeFeature:
    @staticmethod
    def zero_crossing_rate(y, frame_length=512, hop_length=256):
        return np.array([[0.02, 0.02, 0.02]], dtype=np.float32)

    @staticmethod
    def spectral_centroid(y, sr=16000, n_fft=512, hop_length=256):
        return np.array([[420.0, 430.0, 425.0]], dtype=np.float32)


class _FakeLibrosa:
    feature = _FakeFeature()

    @staticmethod
    def note_to_hz(note: str) -> float:
        return 65.0 if note == "C2" else 1046.5

    @staticmethod
    def pyin(y, fmin, fmax, sr=16000, frame_length=1024, hop_length=256):
        f0 = np.array([180.0, 182.0, 181.0, 180.5, 181.5], dtype=np.float32)
        voiced = np.array([True, True, True, True, True])
        return f0, voiced, None


def test_sustained_stable_vocal_tone_detects_moaning(monkeypatch):
    monkeypatch.setattr(detector, "librosa", _FakeLibrosa())
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (0.09 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)

    result = detector.detect_acoustic_vocalization(audio, sr)

    assert result.label == "moaning"
    assert result.score >= 0.60
    assert "pitch_median" in result.evidence


def test_quiet_audio_does_not_detect(monkeypatch):
    monkeypatch.setattr(detector, "librosa", _FakeLibrosa())
    audio = np.zeros(16000, dtype=np.float32)

    result = detector.detect_acoustic_vocalization(audio, 16000)

    assert result.label is None
    assert result.score == 0.0

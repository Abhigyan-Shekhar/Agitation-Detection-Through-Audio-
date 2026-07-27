"""Tests for acoustic feature extraction."""
from __future__ import annotations

import time
import numpy as np
import pytest

pytest.importorskip("librosa")

from acoustic_features import AcousticExtractor, _safe, _AudioRecord


def _make_record(
    freq: float = 440.0,
    duration: float = 0.1,
    sr: int = 16000,
    is_speech: bool = True,
) -> _AudioRecord:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return _AudioRecord(data=data, timestamp=time.time(), is_speech=is_speech)


class TestSafeFloat:

    def test_finite_passthrough(self):
        assert _safe(0.5) == pytest.approx(0.5)

    def test_nan_returns_default(self):
        assert _safe(float("nan")) == 0.0

    def test_inf_returns_default(self):
        assert _safe(float("inf")) == 0.0

    def test_custom_default(self):
        assert _safe(float("nan"), default=99.0) == 99.0


class TestAcousticExtractor:
    def setup_method(self):
        self.extractor = AcousticExtractor(sample_rate=16000)

    def test_empty_records_returns_zero_window(self):
        w = self.extractor.extract([], 0.0, 2.0)
        assert w.rms_mean == 0.0
        assert w.pitch_median == 0.0
        assert w.voiced_ratio == 0.0

    def test_sine_wave_nonzero_rms(self):
        records = [_make_record(freq=440.0) for _ in range(20)]
        w = self.extractor.extract(records, 0.0, 2.0)
        assert w.rms_mean > 0.0

    def test_clipping_ratio_silent(self):
        records = [_make_record(freq=440.0) for _ in range(10)]
        w = self.extractor.extract(records, 0.0, 2.0)
        assert w.clipping_ratio == pytest.approx(0.0, abs=0.01)

    def test_clipping_ratio_clipped(self):
        record = _make_record(freq=440.0)
        record = _AudioRecord(
            data=np.ones(len(record.data), dtype=np.float32),
            timestamp=record.timestamp,
            is_speech=True,
        )
        w = self.extractor.extract([record] * 10, 0.0, 2.0)
        assert w.clipping_ratio > 0.9

    def test_voiced_ratio_all_speech(self):
        records = [_make_record(is_speech=True) for _ in range(10)]
        w = self.extractor.extract(records, 0.0, 2.0)
        assert w.voiced_ratio == pytest.approx(1.0)

    def test_voiced_ratio_no_speech(self):
        records = [_make_record(is_speech=False) for _ in range(10)]
        w = self.extractor.extract(records, 0.0, 2.0)
        assert w.voiced_ratio == pytest.approx(0.0)

    def test_pause_ratio_complement(self):
        records = [
            _make_record(is_speech=i % 2 == 0) for i in range(10)
        ]
        w = self.extractor.extract(records, 0.0, 2.0)
        assert w.voiced_ratio + w.pause_ratio == pytest.approx(1.0, abs=1e-6)

    def test_timestamps_preserved(self):
        records = [_make_record() for _ in range(5)]
        w = self.extractor.extract(records, 1.0, 3.0)
        assert w.start_time == 1.0
        assert w.end_time == 3.0

    def test_all_fields_finite(self):
        records = [_make_record(freq=440.0) for _ in range(20)]
        w = self.extractor.extract(records, 0.0, 2.0)
        for field_name in [
            "rms_mean", "rms_max", "rms_slope", "pitch_median",
            "pitch_range", "pitch_variance", "zcr_mean", "spectral_centroid",
            "voiced_ratio", "pause_ratio", "clipping_ratio",
        ]:
            val = getattr(w, field_name)
            assert np.isfinite(val), f"{field_name}={val} is not finite"

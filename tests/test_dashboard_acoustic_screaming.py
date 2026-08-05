from __future__ import annotations

import time

from audio_pipeline import LoudnessSnapshot
from dashboard import _loudness_scream_score


def test_recent_loudness_snapshot_triggers_scream_score():
    loudness = LoudnessSnapshot(
        timestamp=time.time(),
        rms=0.32,
        peak=0.80,
        clipping_ratio=0.0,
        frame_index=1,
    )

    assert _loudness_scream_score(loudness) >= 0.65


def test_moderate_loudness_snapshot_with_strong_peak_triggers_scream_score():
    loudness = LoudnessSnapshot(
        timestamp=time.time(),
        rms=0.16,
        peak=0.70,
        clipping_ratio=0.0,
        frame_index=1,
    )

    assert _loudness_scream_score(loudness) >= 0.65


def test_stale_loudness_snapshot_is_ignored():
    loudness = LoudnessSnapshot(
        timestamp=time.time() - 2.0,
        rms=0.90,
        peak=1.0,
        clipping_ratio=0.50,
        frame_index=1,
    )

    assert _loudness_scream_score(loudness) == 0.0

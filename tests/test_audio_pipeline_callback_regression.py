import queue

import numpy as np
import pytest

from audio_pipeline import AudioPipeline, LoudnessSnapshot, TimestampedFrame


def test_audio_callback_enqueues_frame_without_debug_capture_hook():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=4, max_queue_size=4)
    indata = np.array([[0.01], [-0.02], [0.03], [-0.04]], dtype=np.float32)

    assert pipeline._capture_callback_audio(indata) is None

    pipeline._audio_callback(indata, frames=4, time_info=None, status=None)

    acoustic_frame = pipeline.acoustic_queue.get_nowait()
    transcription_frame = pipeline.transcription_queue.get_nowait()

    assert isinstance(acoustic_frame, TimestampedFrame)
    assert isinstance(transcription_frame, TimestampedFrame)
    np.testing.assert_array_equal(acoustic_frame.data, indata[:, 0])
    np.testing.assert_array_equal(transcription_frame.data, indata[:, 0])
    assert acoustic_frame.frame_index == 1
    assert transcription_frame.frame_index == 1
    assert isinstance(pipeline.latest_loudness, LoudnessSnapshot)
    assert pipeline.latest_loudness.rms > 0.0
    assert pipeline.latest_loudness.peak == pytest.approx(0.04)
    assert pipeline.latest_loudness.frame_index == 1
    assert pipeline.stats["frames_captured"] == 1
    assert pipeline.stats["acoustic_frames_enqueued"] == 1
    assert pipeline.stats["transcription_frames_enqueued"] == 1
    assert pipeline.stats["transcription_queue_depth"] == 0

    try:
        pipeline.acoustic_queue.get_nowait()
    except queue.Empty:
        pass
    else:
        raise AssertionError("callback enqueued more than one acoustic frame")


def test_audio_callback_tracks_loud_clipped_frame():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=4, max_queue_size=4)
    indata = np.array([[1.0], [-1.0], [0.75], [-0.75]], dtype=np.float32)

    pipeline._audio_callback(indata, frames=4, time_info=None, status=None)

    loudness = pipeline.latest_loudness
    assert loudness is not None
    assert loudness.rms >= 0.85
    assert loudness.peak == pytest.approx(1.0)
    assert loudness.clipping_ratio == pytest.approx(0.5)

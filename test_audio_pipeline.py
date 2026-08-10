import numpy as np

from audio_pipeline import AudioPipeline, TimestampedFrame


def test_audio_callback_fans_out_timestamped_frames():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=4, max_queue_size=5)
    indata = np.array([[0.1], [0.2], [-0.1], [0.0]], dtype=np.float32)

    pipeline._audio_callback(indata, frames=4, time_info=None, status=None)

    acoustic_frame = pipeline.acoustic_queue.get_nowait()
    wlk_frame = pipeline.wlk_queue.get_nowait()

    assert isinstance(acoustic_frame, TimestampedFrame)
    assert isinstance(wlk_frame, TimestampedFrame)
    np.testing.assert_array_equal(acoustic_frame.data, indata[:, 0])
    np.testing.assert_array_equal(wlk_frame.data, indata[:, 0])
    assert acoustic_frame.timestamp == wlk_frame.timestamp


def test_audio_callback_drops_frames_when_queues_are_full():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=2, max_queue_size=1)
    indata = np.array([[0.1], [0.2]], dtype=np.float32)

    pipeline._audio_callback(indata, frames=2, time_info=None, status=None)
    pipeline._audio_callback(indata, frames=2, time_info=None, status=None)

    assert pipeline.acoustic_queue.qsize() == 1
    assert pipeline.wlk_queue.qsize() == 1
    assert pipeline.dropped_frames == 2


def test_flush_queues_drains_outputs():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=2, max_queue_size=5)
    indata = np.array([[0.1], [0.2]], dtype=np.float32)
    pipeline._audio_callback(indata, frames=2, time_info=None, status=None)

    pipeline._flush_queues()

    assert pipeline.acoustic_queue.empty()
    assert pipeline.wlk_queue.empty()

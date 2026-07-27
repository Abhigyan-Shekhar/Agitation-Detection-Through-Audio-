import numpy as np

from audio_pipeline import AudioPipeline, TimestampedFrame


def test_audio_callback_fans_out_timestamped_frames_to_both_queues():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=4, max_queue_size=10)
    indata = np.array([[0.1], [-0.2], [0.3], [0.4]], dtype=np.float32)

    pipeline._audio_callback(indata, frames=4, time_info=None, status=None)

    acoustic_frame = pipeline.acoustic_queue.get_nowait()
    wlk_frame = pipeline.wlk_queue.get_nowait()

    assert isinstance(acoustic_frame, TimestampedFrame)
    assert isinstance(wlk_frame, TimestampedFrame)
    assert np.array_equal(acoustic_frame.data, indata[:, 0])
    assert np.array_equal(wlk_frame.data, indata[:, 0])
    assert acoustic_frame.timestamp == wlk_frame.timestamp
    assert pipeline.dropped_frames == 0


def test_audio_callback_drops_frames_when_output_queues_are_full():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=2, max_queue_size=1)
    first = np.array([[0.1], [0.2]], dtype=np.float32)
    second = np.array([[0.3], [0.4]], dtype=np.float32)

    pipeline._audio_callback(first, frames=2, time_info=None, status=None)
    pipeline._audio_callback(second, frames=2, time_info=None, status=None)

    assert pipeline.acoustic_queue.qsize() == 1
    assert pipeline.wlk_queue.qsize() == 1
    assert pipeline.dropped_frames == 2
    assert np.array_equal(pipeline.acoustic_queue.get_nowait().data, first[:, 0])
    assert np.array_equal(pipeline.wlk_queue.get_nowait().data, first[:, 0])


def test_flush_queues_removes_stale_frames():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=2, max_queue_size=10)
    indata = np.array([[0.1], [0.2]], dtype=np.float32)

    pipeline._audio_callback(indata, frames=2, time_info=None, status=None)
    assert pipeline.acoustic_queue.qsize() == 1
    assert pipeline.wlk_queue.qsize() == 1

    pipeline._flush_queues()

    assert pipeline.acoustic_queue.empty()
    assert pipeline.wlk_queue.empty()

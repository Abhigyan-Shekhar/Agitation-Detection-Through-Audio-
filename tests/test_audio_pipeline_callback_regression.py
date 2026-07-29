import queue

import numpy as np

from audio_pipeline import AudioPipeline, TimestampedFrame


def test_audio_callback_enqueues_frame_without_debug_capture_hook():
    pipeline = AudioPipeline(sample_rate=16000, frame_size=4, max_queue_size=4)
    indata = np.array([[0.01], [-0.02], [0.03], [-0.04]], dtype=np.float32)

    pipeline._audio_callback(indata, frames=4, time_info=None, status=None)

    acoustic_frame = pipeline.acoustic_queue.get_nowait()
    wlk_frame = pipeline.wlk_queue.get_nowait()

    assert isinstance(acoustic_frame, TimestampedFrame)
    assert isinstance(wlk_frame, TimestampedFrame)
    np.testing.assert_array_equal(acoustic_frame.data, indata[:, 0])
    np.testing.assert_array_equal(wlk_frame.data, indata[:, 0])
    assert acoustic_frame.frame_index == 1
    assert wlk_frame.frame_index == 1

    try:
        pipeline.acoustic_queue.get_nowait()
    except queue.Empty:
        pass
    else:
        raise AssertionError("callback enqueued more than one acoustic frame")

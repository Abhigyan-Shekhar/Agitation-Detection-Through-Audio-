import threading
import time

import numpy as np
import torch

from audio_pipeline import AudioPipeline, FeatureTranscriptionProcessor


def generate_sine_wave(freq, sample_rate, duration, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)


def test_synthetic_audio_pipeline():
    class FakeWhisperModel:
        def transcribe(self, audio, **kwargs):
            return {
                "text": "",
                "segments": [],
                "info": type("Info", (), {"language": "en"})(),
            }

    pipeline = AudioPipeline(
        window_seconds=1.0,
        overlap_seconds=0.5,
        speech_ratio_threshold=0.3,
        load_vad=False,
        person2_processor=FeatureTranscriptionProcessor(
            sample_rate=16000, whisper_model=FakeWhisperModel()
        ),
    )

    pipeline.is_running = True
    pipeline.vad_model = lambda tensor, sr: torch.tensor([0.9])

    pipeline.process_thread = threading.Thread(
        target=pipeline._process_loop, daemon=True
    )
    pipeline.process_thread.start()

    try:
        silence = np.zeros(int(16000 * 1.5), dtype=np.float32)
        tone = generate_sine_wave(440, 16000, 1.5).astype(np.float32)
        test_signal = np.concatenate([silence, tone])

        frame_size = 512
        for i in range(0, len(test_signal), frame_size):
            frame = test_signal[i : i + frame_size]
            if len(frame) == frame_size:
                pipeline.audio_queue.put(frame)

        deadline = time.monotonic() + 5
        chunks_emitted = []
        while time.monotonic() < deadline:
            while not pipeline.output_queue.empty():
                chunks_emitted.append(pipeline.output_queue.get())
            if chunks_emitted:
                break
            time.sleep(0.1)

        assert len(chunks_emitted) >= 1
        for chunk in chunks_emitted:
            assert "acoustic_features" in chunk
            assert "transcript" in chunk
            assert chunk["transcript"] == ""
            assert isinstance(chunk["acoustic_features"], dict)

    finally:
        pipeline.stop()


def test_person2_processes_speech_and_silence():
    class FakeWhisperModel:
        def transcribe(self, audio, **kwargs):
            return {
                "text": "hello world",
                "segments": [
                    {
                        "text": "hello world",
                        "start": 0.0,
                        "end": 0.5,
                        "confidence": 0.99,
                    }
                ],
                "info": type("Info", (), {"language": "en"})(),
            }

    processor = FeatureTranscriptionProcessor(
        sample_rate=16000, whisper_model=FakeWhisperModel()
    )
    speech = generate_sine_wave(440, 16000, 0.2).astype(np.float32)
    speech_result = processor.process(speech)

    assert speech_result["transcript"] == "hello world"
    assert speech_result["transcript_details"]["segments"][0]["confidence"] == 0.99
    for value in speech_result["acoustic_features"].values():
        assert isinstance(value, float)
        assert np.isfinite(value)

    silence_result = processor.process(np.zeros(16000, dtype=np.float32))
    assert silence_result["transcript"] == "hello world"
    for value in silence_result["acoustic_features"].values():
        assert isinstance(value, float)
        assert np.isfinite(value)


def test_person2_handles_short_audio():
    class FakeWhisperModel:
        def transcribe(self, audio, **kwargs):
            return {
                "text": "",
                "segments": [],
                "info": type("Info", (), {"language": "en"})(),
            }

    processor = FeatureTranscriptionProcessor(
        sample_rate=16000, whisper_model=FakeWhisperModel()
    )
    result = processor.process(np.array([0.1, -0.2], dtype=np.float32))
    assert result["transcript"] == ""
    assert result["transcript_details"]["segments"] == []


if __name__ == "__main__":
    test_synthetic_audio_pipeline()

import queue
import time
import threading
import torch
import numpy as np
import pytest

from audio_pipeline import AudioPipeline

def generate_sine_wave(freq, sample_rate, duration, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * freq * t)

def test_synthetic_audio_pipeline():
    # Setup pipeline with shorter intervals for fast testing
    pipeline = AudioPipeline(window_seconds=1.0, overlap_seconds=0.5, speech_ratio_threshold=0.3)
    
    # Do not start the audio stream, just start the processing thread
    pipeline.is_running = True
    
    # Mock VAD to always return speech probability 0.9 for testing emit logic
    pipeline.vad_model = lambda tensor, sr: torch.tensor([0.9])
    
    pipeline.process_thread = threading.Thread(target=pipeline._process_loop, daemon=True)
    pipeline.process_thread.start()
    
    try:
        # Generate 1.5 seconds of silence
        silence = np.zeros(int(16000 * 1.5), dtype=np.float32)
        
        # Generate 1.5 seconds of sine wave (e.g. 440 Hz)
        # Note: Silero VAD may not perfectly classify a pure sine wave as speech, 
        # but it typically responds to continuous high energy tonal signals.
        # To be safer, we can mix some noise or multiple frequencies, 
        # but let's see if 440Hz works as a basic test.
        tone = generate_sine_wave(440, 16000, 1.5).astype(np.float32)
        
        # Combine
        test_signal = np.concatenate([silence, tone])
        
        # Chunk into 512-sample frames and feed to queue
        frame_size = 512
        for i in range(0, len(test_signal), frame_size):
            frame = test_signal[i:i+frame_size]
            if len(frame) == frame_size:
                pipeline.audio_queue.put(frame)
        
        # Wait for processing
        time.sleep(2)
        
        # Check output queue
        chunks_emitted = []
        while not pipeline.output_queue.empty():
            chunks_emitted.append(pipeline.output_queue.get())
        
        # We sent 3.0 seconds of audio. With window=1.0 and overlap=0.5,
        # we expect evaluations at 1.0s, 1.5s, 2.0s, 2.5s, 3.0s.
        # First 1.5s is silence -> should be discarded.
        # Second 1.5s is tone -> should be emitted (if VAD triggers).
        print(f"Emitted {len(chunks_emitted)} chunks.")
        for chunk in chunks_emitted:
            print(f"Chunk speech ratio: {chunk['speech_ratio']:.2f}")
        
    finally:
        pipeline.stop()

if __name__ == "__main__":
    test_synthetic_audio_pipeline()

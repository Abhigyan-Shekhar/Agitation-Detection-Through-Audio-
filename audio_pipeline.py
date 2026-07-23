import queue
import threading
import time
import numpy as np
import sounddevice as sd
import torch

class AudioPipeline:
    def __init__(self, 
                 sample_rate=16000, 
                 frame_size=512, 
                 window_seconds=5.0, 
                 overlap_seconds=2.5,
                 vad_threshold=0.5,
                 speech_ratio_threshold=0.3):
        
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        
        # Calculate frame counts
        self.window_frames = int((window_seconds * sample_rate) / frame_size)
        self.overlap_frames = int((overlap_seconds * sample_rate) / frame_size)
        self.emit_interval_frames = self.window_frames - self.overlap_frames
        
        self.vad_threshold = vad_threshold
        self.speech_ratio_threshold = speech_ratio_threshold
        
        # Queues
        self.audio_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Buffers
        self.frame_buffer = []       # Stores recent audio frames
        self.vad_buffer = []         # Stores VAD probabilities
        
        # State
        self.is_running = False
        self.process_thread = None
        self.stream = None
        self.frames_since_last_emit = 0
        
        # Load Silero VAD
        # Uses torch.hub to download/load the model.
        print("Loading Silero VAD model...")
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=False,
                                               onnx=False)
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        # Reset model state just in case
        self.vad_model.reset_states()
        print("Silero VAD loaded.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio Callback Status: {status}")
        
        # We need to ensure we only get exactly 'frame_size' samples.
        # indata shape is (frames, channels). We flatten it to 1D.
        audio_data = indata.flatten()
        self.audio_queue.put(audio_data.copy())

    def _preprocess_frame(self, frame):
        """Apply DC offset removal."""
        # frame is float32 in range [-1.0, 1.0] from sounddevice
        # DC offset removal
        frame = frame - np.mean(frame)
        return frame

    def _process_loop(self):
        while self.is_running:
            try:
                # Block with timeout so we can exit cleanly
                frame = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Preprocess
            frame = self._preprocess_frame(frame)
            
            # Convert to torch tensor for VAD
            tensor_frame = torch.from_numpy(frame).float()
            
            # Run VAD
            # Note: We must not run VAD on empty or extremely small frames. 
            # Silero expects batch_size x seq_len. 
            # For streaming, we pass 1D tensor: tensor_frame
            with torch.no_grad():
                speech_prob = self.vad_model(tensor_frame, self.sample_rate).item()
            is_speech = speech_prob >= self.vad_threshold
            
            # Add to buffers
            self.frame_buffer.append(frame)
            self.vad_buffer.append(is_speech)
            
            # Maintain window size
            if len(self.frame_buffer) > self.window_frames:
                self.frame_buffer.pop(0)
                self.vad_buffer.pop(0)
            
            self.frames_since_last_emit += 1
            
            # Check if it's time to emit a chunk
            if self.frames_since_last_emit >= self.emit_interval_frames and len(self.frame_buffer) == self.window_frames:
                self._evaluate_and_emit()
                self.frames_since_last_emit = 0

    def _evaluate_and_emit(self):
        """Evaluate the current 5-second window and emit if sufficient speech."""
        speech_frames = sum(self.vad_buffer)
        speech_ratio = speech_frames / self.window_frames
        
        chunk_data = np.concatenate(self.frame_buffer)
        
        # We also might want to normalize the entire 5-second chunk before emission
        max_val = np.max(np.abs(chunk_data))
        if max_val > 0:
            normalized_chunk = chunk_data / max_val
        else:
            normalized_chunk = chunk_data
            
        if speech_ratio >= self.speech_ratio_threshold:
            print(f"Emitting chunk! Speech ratio: {speech_ratio:.2f}")
            self.output_queue.put({
                "audio": normalized_chunk,
                "speech_ratio": speech_ratio,
                "timestamp": time.time(),
                "duration": len(normalized_chunk) / self.sample_rate
            })
        else:
            print(f"Discarding chunk (speech ratio {speech_ratio:.2f} < {self.speech_ratio_threshold})")

    def start(self):
        if self.is_running:
            return
            
        self.is_running = True
        
        # Reset state
        self.frame_buffer.clear()
        self.vad_buffer.clear()
        self.frames_since_last_emit = 0
        self.vad_model.reset_states()
        
        while not self.audio_queue.empty():
            self.audio_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()
            
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.frame_size,
            dtype='float32',
            callback=self._audio_callback
        )
        self.stream.start()
        print("Audio pipeline started.")

    def stop(self):
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        if self.process_thread:
            self.process_thread.join()
            
        print("Audio pipeline stopped.")

if __name__ == "__main__":
    # Simple manual test
    pipeline = AudioPipeline()
    try:
        pipeline.start()
        print("Listening for 15 seconds... Speak into your microphone!")
        start_time = time.time()
        while time.time() - start_time < 15:
            try:
                chunk = pipeline.output_queue.get(timeout=0.5)
                print(f"Main Thread received chunk: {chunk['duration']}s, speech_ratio: {chunk['speech_ratio']:.2f}")
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.stop()

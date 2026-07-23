import logging
import queue
import threading
import time

import librosa
import numpy as np
import sounddevice as sd
import torch


logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - optional runtime dependency
    WhisperModel = None  # type: ignore[assignment]


class FeatureTranscriptionProcessor:
    def __init__(
        self,
        sample_rate=16000,
        whisper_model=None,
        whisper_model_factory=None,
        model_size="tiny",
        device="cpu",
        compute_type="int8",
        language=None,
    ):
        self.sample_rate = sample_rate
        self._whisper_model = whisper_model
        self._whisper_model_factory = whisper_model_factory
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language

    def _sanitize_finite_float(self, value, default=0.0):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(number):
            return default
        return number

    def extract_acoustic_features(self, audio_segment, sample_rate=None):
        audio = np.asarray(audio_segment, dtype=np.float32)
        sample_rate = self.sample_rate if sample_rate is None else sample_rate

        if audio.size == 0 or audio.size < 4:
            return {
                "rms_energy": 0.0,
                "pitch_mean": 0.0,
                "pitch_variance": 0.0,
                "zero_crossing_rate": 0.0,
                "spectral_centroid": 0.0,
            }

        rms_energy = float(np.mean(librosa.feature.rms(y=audio)))
        zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        spectral_centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
        )

        f0, voiced_flag, _ = librosa.pyin(audio, fmin=50.0, fmax=400.0, sr=sample_rate)
        voiced_f0 = f0[voiced_flag & np.isfinite(f0)]
        if voiced_f0.size == 0:
            pitch_mean = 0.0
            pitch_variance = 0.0
        else:
            pitch_mean = float(np.mean(voiced_f0))
            pitch_variance = float(np.var(voiced_f0))

        return {
            "rms_energy": self._sanitize_finite_float(rms_energy),
            "pitch_mean": self._sanitize_finite_float(pitch_mean),
            "pitch_variance": self._sanitize_finite_float(pitch_variance),
            "zero_crossing_rate": self._sanitize_finite_float(zero_crossing_rate),
            "spectral_centroid": self._sanitize_finite_float(spectral_centroid),
        }

    def _get_whisper_model(self):
        if self._whisper_model is not None:
            return self._whisper_model

        if self._whisper_model_factory is not None:
            self._whisper_model = self._whisper_model_factory()
            return self._whisper_model

        if WhisperModel is None:
            raise ImportError("faster-whisper is not installed")

        self._whisper_model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        return self._whisper_model

    def transcribe(self, audio_segment, sample_rate=None):
        audio = np.asarray(audio_segment, dtype=np.float32)
        if audio.size == 0:
            return {
                "text": "",
                "segments": [],
                "language": self._language,
            }

        try:
            model = self._get_whisper_model()
        except ImportError:
            return {
                "text": "",
                "segments": [],
                "language": self._language,
            }

        if hasattr(model, "transcribe"):
            try:
                result = model.transcribe(
                    audio, language=self._language, beam_size=1, vad_filter=False
                )
            except TypeError:
                result = model.transcribe(audio, language=self._language)

            if isinstance(result, tuple):
                segments, info = result
            elif isinstance(result, dict):
                segments = result.get("segments", [])
                info = result.get("info")
                text = result.get("text", "")
                return {
                    "text": text,
                    "segments": self._normalize_segments(segments),
                    "language": getattr(info, "language", self._language),
                }
            else:
                segments = []
                info = None

            return {
                "text": self._extract_text(segments),
                "segments": self._normalize_segments(segments),
                "language": getattr(info, "language", self._language),
            }

        return {
            "text": "",
            "segments": [],
            "language": self._language,
        }

    def _normalize_segments(self, segments):
        normalized = []
        for segment in segments:
            if isinstance(segment, dict):
                normalized.append(segment)
                continue

            normalized.append(
                {
                    "text": getattr(segment, "text", ""),
                    "start": self._sanitize_finite_float(
                        getattr(segment, "start", 0.0)
                    ),
                    "end": self._sanitize_finite_float(getattr(segment, "end", 0.0)),
                    "confidence": self._confidence_from_segment(segment),
                }
            )
        return normalized

    def _extract_text(self, segments):
        if not segments:
            return ""
        parts = []
        for segment in segments:
            if isinstance(segment, dict):
                text = segment.get("text", "")
            else:
                text = getattr(segment, "text", "")
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    def _confidence_from_segment(self, segment):
        confidence = getattr(segment, "avg_log_prob", None)
        if confidence is None:
            return None
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(confidence):
            return None
        return float(np.exp(confidence))

    def process(self, audio_segment, sample_rate=None):
        features = self.extract_acoustic_features(
            audio_segment, sample_rate=sample_rate
        )
        transcription = self.transcribe(audio_segment, sample_rate=sample_rate)
        return {
            "acoustic_features": features,
            "transcript": transcription.get("text", ""),
            "transcript_details": transcription,
        }


class AudioPipeline:
    def __init__(
        self,
        sample_rate=16000,
        frame_size=512,
        window_seconds=5.0,
        overlap_seconds=2.5,
        vad_threshold=0.5,
        speech_ratio_threshold=0.3,
        load_vad=True,
        person2_processor=None,
    ):

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
        self.frame_buffer = []
        self.vad_buffer = []

        # State
        self.is_running = False
        self.process_thread = None
        self.stream = None
        self.frames_since_last_emit = 0

        self.person2_processor = person2_processor or FeatureTranscriptionProcessor(
            sample_rate=self.sample_rate
        )
        self.vad_model = None
        self.get_speech_timestamps = None
        self.save_audio = None
        self.read_audio = None
        self.VADIterator = None
        self.collect_chunks = None

        if load_vad:
            self._load_vad_model()

    def _load_vad_model(self):
        if torch is None:
            self.vad_model = None
            return

        print("Loading Silero VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils

        self.vad_model.reset_states()
        print("Silero VAD loaded.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio Callback Status: {status}")

        audio_data = indata.flatten()
        self.audio_queue.put(audio_data.copy())

    def _preprocess_frame(self, frame):
        """Apply DC offset removal."""
        frame = frame - np.mean(frame)
        return frame

    def _run_vad(self, frame):
        if self.vad_model is None:
            return np.mean(np.abs(frame)) > 1e-4

        if torch is None:
            speech_prob = self.vad_model(frame, self.sample_rate)
            if hasattr(speech_prob, "item"):
                return speech_prob.item() >= self.vad_threshold
            return float(speech_prob) >= self.vad_threshold

        tensor_frame = torch.from_numpy(frame).float()
        with torch.no_grad():
            speech_prob = self.vad_model(tensor_frame, self.sample_rate).item()
        return speech_prob >= self.vad_threshold

    def _process_loop(self):
        while self.is_running:
            try:
                frame = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Preserve original frame for the buffer to prevent jagged 31Hz discontinuities
            # when the 5-second chunk is later concatenated.
            original_frame = frame
            
            frame = self._preprocess_frame(frame)
            is_speech = self._run_vad(frame)

            self.frame_buffer.append(original_frame)
            self.vad_buffer.append(is_speech)

            if len(self.frame_buffer) > self.window_frames:
                self.frame_buffer.pop(0)
                self.vad_buffer.pop(0)

            self.frames_since_last_emit += 1

            if (
                self.frames_since_last_emit >= self.emit_interval_frames
                and len(self.frame_buffer) == self.window_frames
            ):
                self._evaluate_and_emit()
                self.frames_since_last_emit = 0

    def _evaluate_and_emit(self):
        speech_frames = sum(self.vad_buffer)
        speech_ratio = speech_frames / self.window_frames

        chunk_data = np.concatenate(self.frame_buffer)
        
        # Remove DC offset cleanly across the entire 5-second chunk
        chunk_data = chunk_data - np.mean(chunk_data)

        # Do not dynamically peak-normalize the chunk. Peak normalizing 
        # background noise to 1.0 causes Whisper to hallucinate and ruins 
        # true RMS energy measurements for agitation detection.
        normalized_chunk = chunk_data

        if speech_ratio >= self.speech_ratio_threshold:
            print(f"Emitting chunk! Speech ratio: {speech_ratio:.2f}")
            logger.info("Speech window accepted; starting feature and transcription processing")
            try:
                person2_output = self.person2_processor.process(
                    normalized_chunk, sample_rate=self.sample_rate
                )
            except Exception:
                # Without this boundary an exception terminates the daemon worker
                # and makes the dashboard appear to be waiting indefinitely.
                logger.exception("Feature/transcription processing failed; chunk was not queued")
                return

            logger.info("Feature/transcription processing completed; enqueueing result")
            output = {
                "audio": normalized_chunk,
                "speech_ratio": speech_ratio,
                "timestamp": time.time(),
                "duration": len(normalized_chunk) / self.sample_rate,
                "acoustic_features": person2_output["acoustic_features"],
                "transcript": person2_output["transcript"],
                "transcript_details": person2_output["transcript_details"],
            }
            self.output_queue.put(output)
            logger.info("Output queue put reached; queue size is now %s", self.output_queue.qsize())
        else:
            print(
                f"Discarding chunk (speech ratio {speech_ratio:.2f} < {self.speech_ratio_threshold})"
            )

    def start(self):
        if self.is_running:
            return

        self.is_running = True

        self.frame_buffer.clear()
        self.vad_buffer.clear()
        self.frames_since_last_emit = 0
        if self.vad_model is not None and hasattr(self.vad_model, "reset_states"):
            self.vad_model.reset_states()

        while not self.audio_queue.empty():
            self.audio_queue.get()
        while not self.output_queue.empty():
            self.output_queue.get()

        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

        if sd is None:
            raise RuntimeError("sounddevice is not installed")

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.frame_size,
            dtype="float32",
            callback=self._audio_callback,
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
    pipeline = AudioPipeline()
    try:
        pipeline.start()
        print("Listening for 15 seconds... Speak into your microphone!")
        start_time = time.time()
        while time.time() - start_time < 15:
            try:
                chunk = pipeline.output_queue.get(timeout=0.5)
                print(
                    f"Main Thread received chunk: {chunk['duration']}s, speech_ratio: {chunk['speech_ratio']:.2f}"
                )
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.stop()

import pyaudio
import numpy as np
import queue
import librosa
import torch
import scipy.signal as signal

class MicStream:
    def __init__(self, target_sample_rate=16000, input_device_index=0):
        self.target_sample_rate = target_sample_rate
        self.pa = pyaudio.PyAudio()
        self.device_sample_rate = 48000
        
        # Audio Allocation & Frame Sizing
        # 512 frames at 16kHz = 32ms. At 48kHz, 32ms is 1536 frames.
        self.device_chunk_size = int(self.device_sample_rate * (512 / 16000.0))
        
        # Endpoint Configuration
        self.silence_threshold = 0.35  # VAD confidence below this = silence
        self.consecutive_silence_limit = int(1.5 / 0.032)  # 1.5 seconds worth of 32ms frames (~46 frames)
        self.silence_counter = 0
        self.recording_started = False
        self.is_endpointed = False
        
        # State Arrays
        self.session_chunks = []
        self.audio_queue = queue.Queue()
        self.stream = None

        # Load Silero VAD Offline
        print("[Audio] Loading Silero VAD from local disk...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='/home/spark2/Models/silero-vad-master',
            model='silero_vad',
            source='local',
            onnx=False
        )
        self.vad_model.eval()

    def start(self):
        self.session_chunks = []
        self.silence_counter = 0
        self.recording_started = False
        self.is_endpointed = False
        
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.device_sample_rate,
            input=True,
            frames_per_buffer=self.device_chunk_size,
            stream_callback=self._callback
        )
        self.stream.start_stream()
        print("[Audio] Microphone stream active. Speak now...")

    def pause_listening(self):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            print("\n[Audio] Mic paused during AI processing.")

    def resume_listening(self):
        self.session_chunks = []
        self.silence_counter = 0
        self.recording_started = False
        self.is_endpointed = False
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        if self.stream and self.stream.is_stopped():
            self.stream.start_stream()
            print("[Audio] Mic resumed. Ready for next turn.")

    def _callback(self, in_data, frame_count, time_info, status):
        audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Downsample to 16kHz safely
        resampled_float = librosa.resample(
            audio_float, orig_sr=self.device_sample_rate, target_sr=self.target_sample_rate
        )
        resampled_float = self.gentle_noise_reduction(resampled_float)

        # Enforce exact 512 frame padding for Silero window
        if len(resampled_float) > 512:
            resampled_float = resampled_float[:512]
        elif len(resampled_float) < 512:
            resampled_float = np.pad(resampled_float, (0, 512 - len(resampled_float)))

        # VAD Verification
        with torch.no_grad():
            audio_tensor = torch.from_numpy(resampled_float).float()
            speech_prob = self.vad_model(audio_tensor, self.target_sample_rate).item()

        # State Tracking Logic
        if speech_prob > 0.45:
            if not self.recording_started:
                self.recording_started = True
                print("\n[Audio] Speech detected, recording transaction...", flush=True)
            self.silence_counter = 0
        else:
            if self.recording_started:
                self.silence_counter += 1

        # Keep accumulating audio once speech has initially begun
        if self.recording_started:
            self.session_chunks.append(resampled_float)
            print(".", end="", flush=True)

        # Check Endpoint conditions
        if self.recording_started and self.silence_counter >= self.consecutive_silence_limit:
            self.is_endpointed = True
            # Build final array and push to master processing queue
            raw_session = np.concatenate(self.session_chunks)
            self.audio_queue.put(raw_session)
            self.recording_started = False

        return (in_data, pyaudio.paContinue)

    def gentle_noise_reduction(self, audio_chunk):
        if np.all(audio_chunk == 0):
            return audio_chunk
        try:
            return signal.wiener(audio_chunk, mysize=3)
        except:
            return audio_chunk

    def get_complete_utterance(self):
        """Returns complete audio block only when the user is completely done talking."""
        if self.is_endpointed and not self.audio_queue.empty():
            return self.audio_queue.get()
        return None

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()

import pyaudio
import numpy as np
import queue
import librosa
import torch
import scipy.signal as signal

class MicStream:
    def __init__(self, target_sample_rate, target_chunk_size, input_device_index=0):
        self.target_sample_rate = target_sample_rate  # Gemma/Whisper expect 16000
        self.pa = pyaudio.PyAudio()
        
        # DYNAMIC SCANNER: Search for the real TAE1159 / USB Audio device index
        self.input_device_index = None
        for i in range(self.pa.get_device_count()):
            dev_info = self.pa.get_device_info_by_index(i)
            dev_name = dev_info.get('name', '').lower()
            if 'tae1159' in dev_name or 'usb audio' in dev_name:
                self.input_device_index = i
                print(f"[Audio] Found targeted hardware device at Index ({i}): {dev_info['name']}")
                break
        
        if self.input_device_index is None:
            self.input_device_index = 0
            print("[Audio] Target string match failed. Falling back to default Index 0.")
            
        self.device_sample_rate = 48000
        self.device_chunk_size = int(self.device_sample_rate * (512 / 16000.0))
        
        print("[Audio] Loading Silero VAD from local disk...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='/home/spark2/Models/silero-vad-master',
            model='silero_vad',
            source='local',
            onnx=False
        )
        self.vad_model.eval()
        
        self.stream = None
        self.audio_queue = queue.Queue()
        
        # --- ⏳ VAD STATE TRACKING ARCHITECTURE ---
        self.speech_buffer = []      # Accumulates active audio frames
        self.is_speaking = False     # State flag tracking if user is talking
        self.silence_chunks = 0      # Counts consecutive silent frames
        
        # 512 frames at 16kHz = 32ms per chunk
        # 1.5 seconds of silence = 1.5 / 0.032 = ~47 chunks
        self.max_silence_chunks = 47 
        # 0.5 seconds noise filter = 0.5 / 0.032 = ~16 chunks
        self.min_speech_chunks = 16

    def start(self):
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.device_sample_rate,
            input=True,
            input_device_index=self.input_device_index,
            frames_per_buffer=self.device_chunk_size,
            stream_callback=self._callback
        )
        self.stream.start_stream()
        print("[Audio] Microphone stream started.")

    def pause_listening(self):
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            print("\n[Audio] Mic paused while AI processes...")

    def resume_listening(self):
        if self.stream and self.stream.is_stopped():
            while not self.audio_queue.empty():
                self.audio_queue.get()
            # Reset VAD tracking states to clean clear for new conversational turn
            self.speech_buffer = []
            self.is_speaking = False
            self.silence_chunks = 0
            self.stream.start_stream()
            print("[Audio] Mic resumed. Ready for next input!")

    def _callback(self, in_data, frame_count, time_info, status):
        audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if self.device_sample_rate != self.target_sample_rate:
            resampled_float = librosa.resample(
                audio_float, orig_sr=self.device_sample_rate, target_sr=self.target_sample_rate
            )
        else:
            resampled_float = audio_float
            
        resampled_float = self.gentle_noise_reduction(resampled_float)
        
        # Fixed padding syntax error from original file
        if len(resampled_float) > 512:
            resampled_float = resampled_float[:512]
        elif len(resampled_float) < 512:
            resampled_float = np.pad(resampled_float, (0, 512 - len(resampled_float)))
            
        with torch.no_grad():
            audio_tensor = torch.from_numpy(resampled_float).float()
            self.vad_model.float()
            speech_prob = self.vad_model(audio_tensor, self.target_sample_rate).item()
            
        # --- 🧠 INTELLIGENT ACCUMULATION & ENDPOINTING LOGIC ---
        if speech_prob > 0.45:  # Slightly lowered threshold for sick/weak speech safety
            if not self.is_speaking:
                self.is_speaking = True
                print("\n[VAD] Hearing speech...", end="", flush=True)
            
            self.speech_buffer.append(resampled_float)
            self.silence_chunks = 0
            print(".", end="", flush=True)  # Visual progress dots
        else:
            if self.is_speaking:
                # Keep trailing silence frames inside sentence buffer for natural padding
                self.speech_buffer.append(resampled_float)
                self.silence_chunks += 1
                
                # Check if user hit the 1.5 second silence ceiling
                if self.silence_chunks >= self.max_silence_chunks:
                    # Drop keyboard clicks or bumps shorter than 0.5 seconds
                    if len(self.speech_buffer) >= self.min_speech_chunks:
                        print("\n[VAD] Sentence endpoint reached. Sending to Gemma...")
                        full_phrase = np.concatenate(self.speech_buffer)
                        resampled_int16 = (full_phrase * 32768.0).astype(np.int16)
                        self.audio_queue.put(resampled_int16.tobytes())
                    else:
                        print("\n[VAD] Ignored short noise artifact.")
                        
                    # Reset sentence markers for next exchange
                    self.speech_buffer = []
                    self.is_speaking = False
                    self.silence_chunks = 0
                    
        return (in_data, pyaudio.paContinue)

    def gentle_noise_reduction(self, audio_chunk):
        if np.all(audio_chunk == 0):
            return audio_chunk
        safe_chunk = audio_chunk + 1e-10
        try:
            return signal.wiener(safe_chunk, mysize=3)
        except:
            return audio_chunk

    def get_audio_chunk(self):
        chunks = []
        while not self.audio_queue.empty():
            chunks.append(self.audio_queue.get())
        if chunks:
            raw_data = b''.join(chunks)
            audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            return audio_np
        return None

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pa.terminate()
        print("\n[Audio] Microphone stream stopped.")

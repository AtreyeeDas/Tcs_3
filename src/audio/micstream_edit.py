# --- ⏳ VAD STATE TRACKING ARCHITECTURE ---
        self.speech_buffer = []      # Accumulates active audio frames
        self.is_speaking = False     # State flag tracking if user is talking
        self.silence_chunks = 0      # Counts consecutive silent frames
        
        # ADD THESE TWO LINES HERE:
        self.voiced_chunks = 0       # Tracks actual speech duration to filter clicks
        self.warmup_chunks = 0       # Ignores hardware power-on pops
        
        self.max_silence_chunks = 47 
        self.min_speech_chunks = 16

def resume_listening(self):
        if self.stream and self.stream.is_stopped():
            while not self.audio_queue.empty():
                self.audio_queue.get()
            
            # UPDATE THESE RESETS:
            self.speech_buffer = []
            self.is_speaking = False
            self.silence_chunks = 0
            self.voiced_chunks = 0   
            self.warmup_chunks = 0   
            
            self.stream.start_stream()
            print("[Audio] Mic resumed. Ready for next input!")

  def _callback(self, in_data, frame_count, time_info, status):
        # 1. Hardware Settle Gate: Ignore the first 10 chunks (~320ms) to bypass power-on pops
        if self.warmup_chunks < 10:
            self.warmup_chunks += 1
            return (in_data, pyaudio.paContinue)

        audio_float = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if self.device_sample_rate != self.target_sample_rate:
            resampled_float = librosa.resample(
                audio_float, orig_sr=self.device_sample_rate, target_sr=self.target_sample_rate
            )
        else:
            resampled_float = audio_float
            
        resampled_float = self.gentle_noise_reduction(resampled_float)
        
        if len(resampled_float) > 512:
            resampled_float = resampled_float[:512]
        elif len(resampled_float) < 512:
            resampled_float = np.pad(resampled_float, (0, 512 - len(resampled_float)))
            
        with torch.no_grad():
            audio_tensor = torch.from_numpy(resampled_float).float()
            self.vad_model.float()
            speech_prob = self.vad_model(audio_tensor, self.target_sample_rate).item()
            
        # --- 🧠 INTELLIGENT ACCUMULATION & ENDPOINTING LOGIC ---
        if speech_prob > 0.45:  
            if not self.is_speaking:
                self.is_speaking = True
                print("\n[VAD] Hearing speech...", end="", flush=True)
            
            self.speech_buffer.append(resampled_float)
            self.silence_chunks = 0
            self.voiced_chunks += 1   # Increment active vocalizations
            print(".", end="", flush=True)  
        else:
            if self.is_speaking:
                self.speech_buffer.append(resampled_float)
                self.silence_chunks += 1
                
                if self.silence_chunks >= self.max_silence_chunks:
                    # CRITICAL FIX: Base noise gate on voiced frames, not total buffer array size
                    # 10 chunks requires at least 320ms of true active human speech
                    if self.voiced_chunks >= 10: 
                        print("\n[VAD] Sentence endpoint reached. Sending to Gemma...")
                        full_phrase = np.concatenate(self.speech_buffer)
                        resampled_int16 = (full_phrase * 32768.0).astype(np.int16)
                        self.audio_queue.put(resampled_int16.tobytes())
                    else:
                        print("\n[VAD] Ignored short noise artifact or click.")
                        
                    self.speech_buffer = []
                    self.is_speaking = False
                    self.silence_chunks = 0
                    self.voiced_chunks = 0
                    
        return (in_data, pyaudio.paContinue)

# src/tts/parler_engine.py
import os
import io
import requests
import sounddevice as sd
import soundfile as sf

class ParlerTTSEngine:
    def __init__(self, model_path, voice_description, output_dir, device="cuda"):
        # We no longer load the model here! We just talk to localhost.
        self.output_dir = output_dir
        self.voice_description = voice_description
        self.server_url = "http://127.0.0.1:8000/generate"
        print("Connected to Parler-TTS Microservice.")

    def synthesize_and_play(self, text):
        if not text:
            return
            
        print(f"[TTS Client] Requesting audio for: {text[:30]}...")
        
        # 1. Ask the microservice to generate the audio
        payload = {
            "text": text,
            "voice_description": self.voice_description
        }
        
        try:
            response = requests.post(self.server_url, json=payload)
            response.raise_for_status() # Check for errors
            
            # 2. Read the audio bytes sent back from the server
            audio_data, sample_rate = sf.read(io.BytesIO(response.content))
            
            # 3. Save for auditing and play it aloud
            output_file = os.path.join(self.output_dir, "response.wav")
            sf.write(output_file, audio_data, sample_rate)
            
            sd.play(audio_data, sample_rate)
            sd.wait()
            
        except requests.exceptions.ConnectionError:
            print("[Error] TTS Server is not running! Did you start tts_server.py?")

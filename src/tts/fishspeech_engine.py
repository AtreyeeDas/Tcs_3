import os
import sounddevice as sd
import soundfile as sf
# Standard imports for FishSpeech v1.5 inference
from fish_speech.inference_engine import FishSpeechTTS 

class FishSpeechEngine:
    def __init__(self, model_path, reference_wav, output_dir):
        print(f"Loading FishSpeech TTS from {model_path}...")
        self.output_dir = output_dir
        
        # Initialize the modern FishSpeech engine
        self.tts = FishSpeechTTS(
            model_path=model_path,
            device="cuda",
            half_precision=True # Utilizes your Blackwell architecture efficiently
        )
        
        # Pre-load the reference voice for zero-shot cloning
        print(f"Loading reference voice: {reference_wav}")
        self.voice_prompt = self.tts.load_reference_audio(reference_wav)
        
    def synthesize_and_play(self, text):
        if not text:
            return
            
        output_file = os.path.join(self.output_dir, "response.wav")
        
        # Generate the audio array using the reference voice
        audio_array, sample_rate = self.tts.synthesize(
            text=text,
            reference_audio=self.voice_prompt,
            language="auto" # Let FishSpeech automatically handle English/Hindi switching
        )
        
        # Save audio for auditing
        sf.write(output_file, audio_array, sample_rate)
        
        # Play the audio back to the user
        sd.play(audio_array, sample_rate)
        sd.wait() # Block execution until audio finishes playing

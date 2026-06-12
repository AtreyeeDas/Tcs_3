import os
import torch
import sounddevice as sd
import soundfile as sf
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

class ParlerTTSEngine:
    def __init__(self, model_path, voice_description, output_dir, device="cuda"):
        print(f"Loading Indic Parler-TTS from {model_path}...")
        self.output_dir = output_dir
        self.device = device
        self.voice_description = voice_description
        
        # Must use the specific Parler class to ensure the audio decoder loads correctly
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.sample_rate = self.model.config.sampling_rate

    def synthesize_and_play(self, text):
        if not text:
            return
            
        output_file = os.path.join(self.output_dir, "response.wav")
        
        # Parler terminology is slightly backwards:
        # 'input_ids' = the description of the voice
        # 'prompt_input_ids' = the actual transcript text you want spoken
        description_tokens = self.tokenizer(self.voice_description, return_tensors="pt").input_ids.to(self.device)
        transcript_tokens = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        with torch.no_grad():
            generation = self.model.generate(
                input_ids=description_tokens,
                prompt_input_ids=transcript_tokens
            )
        
        # Extract the audio array (squeeze to 1D)
        audio_array = generation.cpu().numpy().squeeze()
        
        # Save for auditing and play to the user
        sf.write(output_file, audio_array, self.sample_rate)
        sd.play(audio_array, self.sample_rate)
        sd.wait()

# 2. Unified Native Audio-to-Text inference
print("\n[AI]: Listening and Reasoning...")
transcription, response = self.multimodal.generate_response(
    audio_array=audio_array, 
    system_prompt=Config.MEDICAL_PROMPT
)

# Print both outputs to the terminal as requested
print("-" * 50)
print(f"[Patient Said]: {transcription}")
print(f"[Doctor Replied]: {response}")
print("-" * 50)

# 3. Native TTS Synthesis using FishSpeech
print("\n[TTS]: Generating voice...")
self.tts.synthesize_and_play(response) # ONLY pass the response to TTS

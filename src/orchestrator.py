import time
from src.config import Config
from src.audio.mic_stream import MicStream
from src.multimodal.gemma_engine import GemmaAudioEngine
from src.tts.fishspeech_engine import FishSpeechEngine

class PipelineOrchestrator:
    def __init__(self):
        Config.setup_dirs()
        self.mic = MicStream(Config.SAMPLE_RATE, Config.CHUNK_SIZE, Config.INPUT_DEVICE_INDEX)
        
        print("Loading Gemma 4 E4B Multimodal Engine...")
        self.multimodal = GemmaAudioEngine(Config.GEMMA_MODEL_PATH, Config.DEVICE, Config.COMPUTE_TYPE)
        
        print("Loading FishSpeech v1.5 TTS Engine...")
        self.tts = FishSpeechEngine(Config.TTS_MODEL_PATH, Config.SPEAKER_REFERENCE_WAV, Config.TTS_OUTPUT_DIR)

    def run(self):
        self.mic.start()
        print("\n=== Pipeline Active. Start speaking. Press Ctrl+C to stop. ===")
        try:
            while True:
                time.sleep(0.01)
                
                # We assume get_audio_chunk() returns a complete 16kHz numpy array 
                # once the Silero VAD 1.5s silence threshold is triggered.
                audio_array = self.mic.get_audio_chunk()
                
                if audio_array is not None and len(audio_array) > 0:
                    # 1. Pause mic so it doesn't hear the TTS output looping back
                    self.mic.pause_listening()
                    
                    # 2. Unified Native Audio-to-Text inference
                    print("\n[AI]: Reasoning over audio intent...")
                    llm_response = self.multimodal.generate_response(
                        audio_array=audio_array, 
                        system_prompt=Config.MEDICAL_PROMPT
                    )
                    
                    print(f"\n[Output]: {llm_response}")
                    
                    # 3. Native TTS Synthesis using FishSpeech
                    print("\n[TTS]: Generating voice...")
                    self.tts.synthesize_and_play(llm_response)
                    
                    # 4. Wake mic up for next turn
                    self.mic.resume_listening()
                    print("\n=== Ready for your next sentence. ===")
                    
        except KeyboardInterrupt:
            print("\nShutting down pipeline...")
            self.mic.stop()

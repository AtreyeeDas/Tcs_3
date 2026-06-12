class PipelineOrchestrator:
    def __init__(self):
        Config.setup_dirs()
        self.mic = MicStream(Config.SAMPLE_RATE, Config.CHUNK_SIZE, Config.INPUT_DEVICE_INDEX)
        
        print("Loading Gemma 4 E4B Multimodal Engine...")
        self.multimodal = GemmaAudioEngine(Config.GEMMA_MODEL_PATH, Config.DEVICE, Config.COMPUTE_TYPE)
        
        # --- Plug and Play TTS Router ---
        print(f"Loading {Config.ACTIVE_TTS_ENGINE.upper()} TTS Engine...")
        if Config.ACTIVE_TTS_ENGINE == "fishspeech":
            from src.tts.fishspeech_engine import FishSpeechEngine
            self.tts = FishSpeechEngine(
                Config.FISH_MODEL_PATH, 
                Config.SPEAKER_REFERENCE_WAV, 
                Config.TTS_OUTPUT_DIR
            )
        elif Config.ACTIVE_TTS_ENGINE == "parler":
            from src.tts.parler_engine import ParlerTTSEngine
            self.tts = ParlerTTSEngine(
                Config.PARLER_MODEL_PATH, 
                Config.PARLER_VOICE_PROMPT, 
                Config.TTS_OUTPUT_DIR
            )

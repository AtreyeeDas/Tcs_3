class Config:
    # ... [Keep your existing Gemma and Audio Settings] ...

    # --- 🔀 PLUG AND PLAY TTS TOGGLE ---
    ACTIVE_TTS_ENGINE = "parler" # Change to "fishspeech" to swap engines

    # --- FishSpeech Variables ---
    FISH_MODEL_PATH = "/home/spark2/Models/FishSpeech-1.5"
    SPEAKER_REFERENCE_WAV = "/home/spark2/Models/doctor_voice.wav"

    # --- Parler-TTS Variables ---
    PARLER_MODEL_PATH = "ai4bharat/indic-parler-tts"
    # Parler requires a text prompt to generate the voice instead of a .wav
    PARLER_VOICE_PROMPT = (
        "A male doctor speaks in a calm, authoritative, and deeply empathetic voice "
        "with a clear Indian accent at a moderate pace. The recording is of very high quality "
        "with absolutely no background noise."
    )

    TTS_OUTPUT_DIR = "./output_audio"

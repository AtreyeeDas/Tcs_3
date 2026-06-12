import os

class Config:
    # Audio Settings
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 480
    CHANNELS = 2
    INPUT_DEVICE_INDEX = 0

    # Environment & Hardware Settings
    DEVICE = "cuda"
    COMPUTE_TYPE = "bfloat16" # Optimal for Blackwell / Gemma 4

    # Medical Prompt (Gemma 4 System Prompt)
    MEDICAL_PROMPT = (
        "<|think|> You are an expert clinical cardiologist assistant. "
        "Listen to the patient's audio intent and respond directly with clinical, "
        "empathetic advice in text. Keep the response concise."
    )

    # Multimodal Engine Settings
    GEMMA_MODEL_PATH = "/home/spark2/Models/gemma-4-E4B-it" # Update path if needed

    # TTS Settings
    TTS_MODEL_PATH = "/home/spark2/Models/FishSpeech-1.5" # Update path if needed
    SPEAKER_REFERENCE_WAV = "/home/spark2/Models/doctor_voice.wav"
    TTS_OUTPUT_DIR = "./output_audio"

    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.TTS_OUTPUT_DIR, exist_ok=True)

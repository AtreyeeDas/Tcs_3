# src/tts/tts_server.py
import io
import torch
import soundfile as sf
from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

app = FastAPI()

# 1. Load Parler locally at startup
MODEL_PATH = "/home/spark2/Models/indic-parler-tts"
DEVICE = "cuda"

print("Booting Parler-TTS Microservice...")
model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
sample_rate = model.config.sampling_rate
print("Parler-TTS Server Ready on port 8000!")

# 2. Define the expected data format
class TTSRequest(BaseModel):
    text: str
    voice_description: str

# 3. The Generation Endpoint
@app.post("/generate")
def generate_audio(req: TTSRequest):
    description_tokens = tokenizer(req.voice_description, return_tensors="pt").input_ids.to(DEVICE)
    transcript_tokens = tokenizer(req.text, return_tensors="pt").input_ids.to(DEVICE)

    with torch.no_grad():
        generation = model.generate(
            input_ids=description_tokens,
            prompt_input_ids=transcript_tokens
        )
    
    audio_array = generation.cpu().numpy().squeeze()
    
    # Convert numpy array to WAV bytes in memory (no disk writing needed here)
    wav_io = io.BytesIO()
    sf.write(wav_io, audio_array, sample_rate, format='WAV')
    wav_io.seek(0)
    
    # Send the audio file straight back to Gemma
    return Response(content=wav_io.read(), media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

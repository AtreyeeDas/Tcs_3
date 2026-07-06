import torch
import re
from threading import Thread
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer

class GemmaAudioEngine:
    def __init__(self, model_path, device="cuda", compute_type="bfloat16"):
        print(f"Loading Gemma 4 Processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"Loading Gemma 4 Model from {model_path} to {device}...")
        self.dtype = torch.bfloat16 if compute_type == "bfloat16" else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True, 
            device_map=device,
            torch_dtype=self.dtype
        )
        self.model.eval()
        self.sample_rate = 16000 # Gemma 4 native audio sample rate
        self.history = [] # holds rolling text turns
        self.device = device

    def generate_response_stream(self, audio_array, system_prompt):
        # 1. Determine Language Anchor from History
        expected_lang = "en"
        if len(self.history) > 0:
             last_response = self.history[-1]["content"][0]["text"]
             if any('\u0900' <= c <= '\u097F' for c in last_response):
                 expected_lang = "hi"
                 
        anchor_rule = "Respond strictly using English text (Latin script)." if expected_lang == "en" else "Respond strictly using Hindi text (Devanagari script)."

        # 2. Build the STRICT Prompt (Notice the <audio> tag at the top!)
        structured_prompt = (
            f"<audio>\n{system_prompt}\n\n"
            "You MUST format your output exactly like this:\n"
            "<Transcription> [Write exactly what the patient said in the audio] </Transcription>\n"
            f"<Language> {expected_lang} </Language>\n"
            f"<Response> [Write your clinical, empathetic response here. {anchor_rule}] </Response>"
        )

        # 3. Process inputs into tensors natively
        inputs = self.processor(text=structured_prompt, audios=audio_array, return_tensors="pt", sampling_rate=self.sample_rate).to(self.device)
        
        # 4. CRITICAL FIX: Setup Streamer with strict prompt skipping
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # 5. Configure Generation (Strict stop parameters)
        generation_kwargs = dict(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=256, 
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=self.processor.tokenizer.eos_token_id # Forces the model to stop when done!
        )
        
        # 6. Launch Thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 7. Yield tokens live
        for new_token in streamer:
            yield new_token

    def update_history(self, transcription, response):
        """Safely updates history using text only."""
        if transcription and response:
            self.history.append({"role":"user","content":[{"type":"text","text":transcription}]})
            self.history.append({"role":"assistant","content":[{"type":"text","text":response}]})
            
            # Keep history short to prevent VRAM explosion
            if len(self.history) > 10:
                self.history = self.history[-10:]

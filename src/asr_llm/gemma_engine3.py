import torch
from threading import Thread
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer

class GemmaAudioEngine:
    def __init__(self, model_path, device="cuda", compute_type="bfloat16"):
        print(f"Loading Multimodal Processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"Loading Multimodal Model from {model_path} to {device}...")
        self.dtype = torch.bfloat16 if compute_type == "bfloat16" else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True, 
            device_map=device,
            torch_dtype=self.dtype
        )
        self.model.eval()
        self.sample_rate = 16000 
        self.history = [] 
        self.device = device 

    def _format_prompt(self, system_prompt):
        expected_lang = "en"
        if len(self.history) > 0:
             last_response = self.history[-1]["content"]
             if any('\u0900' <= c <= '\u097F' for c in last_response):
                 expected_lang = "hi"
                 
        anchor_rule = "Respond strictly in English." if expected_lang == "en" else "Respond strictly in Hindi."

        structured_text = (
            f"{system_prompt}\n\n"
            "Format:\n<Transcription>...</Transcription>\n<Language>...</Language>\n<Response>... " + anchor_rule + "</Response>"
        )
        
        # We return the list structure for apply_chat_template
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio"},
                    {"type": "text", "text": structured_text}
                ]
            }
        ], expected_lang

    def generate_response_stream(self, audio_array, system_prompt):
        # 1. Format and prepare
        messages, lang = self._format_prompt(system_prompt)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 2. FIX: Pre-process audio separately to avoid keyword/ValueError crashes
        # This ensures the processor maps audio to the <audio> tag perfectly.
        inputs = self.processor(
            text=prompt, 
            audio=audio_array, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).to(self.device)
            
        # 3. Setup Streamer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # 4. Configure Generation
        generation_kwargs = dict(
            **inputs, 
            streamer=streamer, 
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.processor.tokenizer.eos_token_id
        )
        
        # 5. Launch
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 6. Yield tokens
        for new_token in streamer:
            yield new_token
            
    def update_history(self, transcription, response):
        if transcription and response:
             self.history.append({"role": "user", "content": transcription})
             self.history.append({"role": "assistant", "content": response})
             if len(self.history) > 10:
                 self.history = self.history[-10:]

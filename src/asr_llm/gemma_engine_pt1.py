import torch
import re
import threading
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
        self.sample_rate = 16000
        self.history = []

    def _detect_script(self, text):
        if not text: return "en"
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        return "hi" if devanagari > latin else "en"

    def generate_response(self, audio_array, system_prompt, sentence_queue=None):
        # -------------------------------------------------------
        # REORDERED XML: Language -> Response -> Transcription
        # -------------------------------------------------------
        structured_prompt = f"""
{system_prompt}

====================================================
VERY IMPORTANT INSTRUCTIONS
====================================================
You are a multilingual AI cardiology assistant.
Determine the language from THIS AUDIO.
If English, reply ONLY in English.
If Hindi, reply ONLY in Hindi using Devanagari.

====================================================
OUTPUT FORMAT (STRICT ORDER)
====================================================
Return EXACTLY:

<Language>
en (or hi)
</Language>

<Response>
doctor response
</Response>

<Transcription>
patient transcript
</Transcription>

Nothing outside these tags.
====================================================
"""
        messages = [{"role": "system", "content": [{"type": "text", "text": structured_prompt}]}]
        messages.extend(self.history[-4:])
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Listen carefully to the patient's audio and answer."},
                {"type": "audio", "audio": {"array": audio_array, "sampling_rate": self.sample_rate}}
            ]
        })

        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt_text, audio=audio_array, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # -------------------------------------------------------
        # BACKGROUND STREAMING GENERATION
        # -------------------------------------------------------
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=256,
            temperature=0.15,
            top_p=0.90,
            repetition_penalty=1.10,
            do_sample=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            streamer=streamer
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # -------------------------------------------------------
        # STATE MACHINE PARSER
        # -------------------------------------------------------
        state = "WAIT_LANGUAGE"
        buffer = ""
        current_sentence = ""
        
        full_response = ""
        full_transcription = ""
        response_lang = "en"  # Default

        for text_chunk in streamer:
            buffer += text_chunk

            if state == "WAIT_LANGUAGE":
                if "</Language>" in buffer:
                    match = re.search(r"<Language>\s*(.*?)\s*</Language>", buffer, re.IGNORECASE | re.DOTALL)
                    if match:
                        lang_tag = match.group(1).strip().lower()
                        response_lang = "hi" if lang_tag in ["hi", "hindi"] else "en"
                    state = "WAIT_RESPONSE"

            elif state == "WAIT_RESPONSE":
                if "<Response>" in buffer:
                    buffer = buffer.split("<Response>")[-1]
                    state = "STREAM_RESPONSE"

            elif state == "STREAM_RESPONSE":
                if "</Response>" in buffer:
                    # Finalize response stream
                    parts = buffer.split("</Response>")
                    chunk = parts[0]
                    full_response += chunk
                    current_sentence += chunk
                    
                    if current_sentence.strip() and sentence_queue is not None:
                        sentence_queue.put((current_sentence.strip(), response_lang))
                    
                    current_sentence = ""
                    buffer = parts[1] if len(parts) > 1 else ""
                    state = "WAIT_TRANSCRIPTION"
                else:
                    # Look for sentence boundaries (. ! ? ।) followed by whitespace
                    matches = list(re.finditer(r'([.!?।]+[\s\n]+)', buffer))
                    if matches:
                        last_match = matches[-1]
                        split_idx = last_match.end()
                        
                        chunk_to_process = buffer[:split_idx]
                        full_response += chunk_to_process
                        current_sentence += chunk_to_process
                        
                        if current_sentence.strip() and sentence_queue is not None:
                            sentence_queue.put((current_sentence.strip(), response_lang))
                        
                        current_sentence = ""
                        buffer = buffer[split_idx:]

            elif state == "WAIT_TRANSCRIPTION":
                if "</Transcription>" in buffer:
                    match = re.search(r"<Transcription>\s*(.*?)\s*</Transcription>", buffer, re.IGNORECASE | re.DOTALL)
                    if match:
                        full_transcription = match.group(1).strip()
                    state = "DONE"

        # Ensure generation is physically complete
        thread.join()

        # -------------------------------------------------------
        # FALLBACKS & STORAGE (Run Exactly Once)
        # -------------------------------------------------------
        if not full_response: full_response = buffer.strip()
        if not full_transcription: full_transcription = "[Gemma did not separate the transcription]"

        full_transcription = " ".join(full_transcription.split())
        full_response = " ".join(full_response.split())

        if full_transcription and full_transcription != "[Gemma did not separate the transcription]":
            self.history.append({"role": "user", "content": [{"type": "text", "text": full_transcription}]})
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": full_response}]})
            if len(self.history) > 8:
                self.history = self.history[-8:]

        return full_transcription, full_response, response_lang

"""
gemma_update_info
import torch
import re
import threading
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
        self.sample_rate = 16000
        self.history = []

    def _detect_script(self, text):
        if not text: return "en"
        devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        latin = sum(1 for c in text if c.isascii() and c.isalpha())
        return "hi" if devanagari > latin else "en"

    def generate_response(self, audio_array, system_prompt, sentence_queue=None):
        # -------------------------------------------------------
        # REORDERED XML: Language -> Response -> Transcription
        # -------------------------------------------------------
        structured_prompt = f"""
{system_prompt}

====================================================
VERY IMPORTANT INSTRUCTIONS
====================================================
You are a multilingual AI cardiology assistant.
Determine the language from THIS AUDIO.
If English, reply ONLY in English.
If Hindi, reply ONLY in Hindi using Devanagari.

====================================================
OUTPUT FORMAT (STRICT ORDER)
====================================================
Return EXACTLY:

<Language>
en (or hi)
</Language>

<Response>
doctor response
</Response>

<Transcription>
patient transcript
</Transcription>

Nothing outside these tags.
====================================================
"""
        messages = [{"role": "system", "content": [{"type": "text", "text": structured_prompt}]}]
        messages.extend(self.history[-4:])
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Listen carefully to the patient's audio and answer."},
                {"type": "audio", "audio": {"array": audio_array, "sampling_rate": self.sample_rate}}
            ]
        })

        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=prompt_text, audio=audio_array, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # -------------------------------------------------------
        # BACKGROUND STREAMING GENERATION
        # -------------------------------------------------------
        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=256,
            temperature=0.15,
            top_p=0.90,
            repetition_penalty=1.10,
            do_sample=True,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            streamer=streamer
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # -------------------------------------------------------
        # STATE MACHINE PARSER
        # -------------------------------------------------------
        state = "WAIT_LANGUAGE"
        buffer = ""
        current_sentence = ""
        
        full_response = ""
        full_transcription = ""
        response_lang = "en"  # Default

        for text_chunk in streamer:
            buffer += text_chunk

            if state == "WAIT_LANGUAGE":
                if "</Language>" in buffer:
                    match = re.search(r"<Language>\s*(.*?)\s*</Language>", buffer, re.IGNORECASE | re.DOTALL)
                    if match:
                        lang_tag = match.group(1).strip().lower()
                        response_lang = "hi" if lang_tag in ["hi", "hindi"] else "en"
                    state = "WAIT_RESPONSE"

            elif state == "WAIT_RESPONSE":
                if "<Response>" in buffer:
                    buffer = buffer.split("<Response>")[-1]
                    state = "STREAM_RESPONSE"

            elif state == "STREAM_RESPONSE":
                if "</Response>" in buffer:
                    # Finalize response stream
                    parts = buffer.split("</Response>")
                    chunk = parts[0]
                    full_response += chunk
                    current_sentence += chunk
                    
                    if current_sentence.strip() and sentence_queue is not None:
                        sentence_queue.put((current_sentence.strip(), response_lang))
                    
                    current_sentence = ""
                    buffer = parts[1] if len(parts) > 1 else ""
                    state = "WAIT_TRANSCRIPTION"
                else:
                    # Look for sentence boundaries (. ! ? ।) followed by whitespace
                    matches = list(re.finditer(r'([.!?।]+[\s\n]+)', buffer))
                    if matches:
                        last_match = matches[-1]
                        split_idx = last_match.end()
                        
                        chunk_to_process = buffer[:split_idx]
                        full_response += chunk_to_process
                        current_sentence += chunk_to_process
                        
                        if current_sentence.strip() and sentence_queue is not None:
                            sentence_queue.put((current_sentence.strip(), response_lang))
                        
                        current_sentence = ""
                        buffer = buffer[split_idx:]

            elif state == "WAIT_TRANSCRIPTION":
                if "</Transcription>" in buffer:
                    match = re.search(r"<Transcription>\s*(.*?)\s*</Transcription>", buffer, re.IGNORECASE | re.DOTALL)
                    if match:
                        full_transcription = match.group(1).strip()
                    state = "DONE"

        # Ensure generation is physically complete
        thread.join()

        # -------------------------------------------------------
        # FALLBACKS & STORAGE (Run Exactly Once)
        # -------------------------------------------------------
        if not full_response: full_response = buffer.strip()
        if not full_transcription: full_transcription = "[Gemma did not separate the transcription]"

        full_transcription = " ".join(full_transcription.split())
        full_response = " ".join(full_response.split())

        if full_transcription and full_transcription != "[Gemma did not separate the transcription]":
            self.history.append({"role": "user", "content": [{"type": "text", "text": full_transcription}]})
            self.history.append({"role": "assistant", "content": [{"type": "text", "text": full_response}]})
            if len(self.history) > 8:
                self.history = self.history[-8:]

        return full_transcription, full_response, response_lang
"""

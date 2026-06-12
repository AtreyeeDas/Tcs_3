import torch
import re
from transformers import AutoProcessor, AutoModelForCausalLM # or AutoModelForImageTextToText depending on HF version

class GemmaAudioEngine:
    def __init__(self, model_path, device="cuda", compute_type="bfloat16"):
        print(f"Loading Gemma 4 Processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"Loading Gemma 4 Model from {model_path} to {device}...")
        self.dtype = torch.bfloat16 if compute_type == "bfloat16" else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=self.dtype
        )
        self.model.eval()
        self.sample_rate = 16000 # Gemma 4 native audio sample rate

    def generate_response(self, audio_array, base_system_prompt):
        # We append strict formatting rules to your Medical Prompt
        structured_prompt = (
            f"{base_system_prompt}\n\n"
            "You MUST format your output exactly like this:\n"
            "<Transcription> [Write exactly what the patient said in the audio] </Transcription>\n"
            "<Response> [Write your clinical, empathetic response] </Response>"
        )

        # Build the native multimodal chat template
        messages = [
            {"role": "system", "content": [{"type": "text", "text": structured_prompt}]},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Listen to the patient's audio:"},
                    {"type": "audio", "audio": {"array": audio_array, "sampling_rate": self.sample_rate}}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )

        # Decode the generated tokens (ignoring the input prompt tokens)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_output = self.processor.decode(generated_tokens, skip_special_tokens=True)

        # Parse the structured tags
        transcription = self._extract_tag(raw_output, "Transcription")
        response = self._extract_tag(raw_output, "Response")

        # Fallback in case the model disobeys instructions
        if not response:
            response = raw_output.strip()
            transcription = "[Gemma did not separate the transcription]"

        return transcription, response

    def _extract_tag(self, text, tag_name):
        match = re.search(f"<{tag_name}>(.*?)</{tag_name}>", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

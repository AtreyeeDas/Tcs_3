import re
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class GemmaAudioEngine:
    def __init__(self, model_path, device="cuda", compute_type="bfloat16"):
        print(f"Loading Gemma 4 Processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        self.dtype = torch.bfloat16 if compute_type == "bfloat16" else torch.float16
        print(f"Loading Gemma 4 Model to {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=self.dtype
        )
        self.model.eval()
        self.sample_rate = 16000
        
        # Persistent System Architecture History Track
        self.history = []

    def generate_response(self, audio_array, system_prompt):
        structured_prompt = (
            f"{system_prompt}\n\n"
            "You MUST format your output exactly like this:\n"
            "<Transcription> [Write exactly what the patient said in the audio] </Transcription>\n"
            "<Response> [Write your clinical, empathetic response] </Response>"
        )

        # Initialize core system context layout if empty
        if len(self.history) == 0:
            self.history.append({"role": "system", "content": [{"type": "text", "text": structured_prompt}]})

        # Append current incoming audio message block
        self.history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Listen to the patient's audio and respond structured:"},
                {"type": "audio", "audio": {"array": audio_array, "sampling_rate": self.sample_rate}}
            ]
        })

        prompt_text = self.processor.apply_chat_template(
            self.history, add_generation_prompt=True, tokenize=False
        )

        inputs = self.processor(
            text=prompt_text, audio=audio_array, sampling_rate=self.sample_rate, return_tensors="pt"
        )

        # Safely map to Blackwell device execution space
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.5, do_sample=True, top_p=0.95
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        raw_output = self.processor.decode(generated_tokens, skip_special_tokens=True)

        transcription = self._extract_tag(raw_output, "Transcription")
        response = self._extract_tag(raw_output, "Response")

        if not response:
            response = raw_output.strip()
            transcription = "[System fell back to text inference extraction manual decode]"

        # Commit the model's generated response back to text memory footprint
        self.history.append({"role": "assistant", "content": [{"type": "text", "text": raw_output}]})

        return transcription, response

    def _extract_tag(self, text, tag_name):
        match = re.search(f"<{tag_name}>(.*?)</{tag_name}>", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

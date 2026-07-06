import torch
import re
from transformers import AutoProcessor, AutoModelForCausalLM


class GemmaAudioEngine:

    def __init__(self, model_path, device="cuda", compute_type="bfloat16"):

        print(f"Loading Gemma 4 Processor from {model_path}...")
        self.processor = AutoProcessor.from_pretrained(model_path)

        print(f"Loading Gemma 4 Model from {model_path} to {device}...")

        self.dtype = (
            torch.bfloat16
            if compute_type == "bfloat16"
            else torch.float16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
            torch_dtype=self.dtype
        )

        self.model.eval()

        self.sample_rate = 16000

        # Rolling conversation history (text only)
        self.history = []


    def _detect_script(self, text):

        if not text:
            return "en"

        devanagari = sum(
            1 for c in text
            if '\u0900' <= c <= '\u097F'
        )

        latin = sum(
            1 for c in text
            if c.isascii() and c.isalpha()
        )

        if devanagari > latin:
            return "hi"

        return "en"


    def generate_response(self, audio_array, system_prompt):

        structured_prompt = f"""
{system_prompt}

====================================================
VERY IMPORTANT INSTRUCTIONS
====================================================

You are a multilingual AI cardiology assistant.

You MUST first determine the language spoken in the
PATIENT AUDIO.

Do NOT use previous conversation language.

Always determine the language from THIS AUDIO.

If the patient's audio is English,
reply ONLY in English.

If the patient's audio is Hindi,
reply ONLY in Hindi using Devanagari.

Never translate the patient's language.

Medical terms like

ECG
MRI
CT Scan
Aspirin
Clopidogrel
Metoprolol
Blood Pressure
Diabetes

may remain in English.

====================================================
OUTPUT FORMAT
====================================================

Return EXACTLY

<Transcription>
patient transcript
</Transcription>

<Response>
doctor response
</Response>

<Language>
en
</Language>

OR

<Language>
hi
</Language>

Nothing outside these tags.

====================================================
SELF VERIFICATION
====================================================

Before producing the final answer verify:

1.
Is the response written in the SAME language as the audio?

2.
Does the Language tag match the response language?

3.
If NOT,
correct yourself BEFORE generating.

====================================================
EXAMPLES
====================================================

Example 1

Patient Audio:
"Doctor I have chest pain."

Output

<Transcription>
Doctor I have chest pain.
</Transcription>

<Response>
Chest pain can sometimes indicate a serious heart condition. If the pain is severe, spreads to your arm, jaw or back, or is accompanied by sweating or shortness of breath, seek emergency medical attention immediately.
</Response>

<Language>
en
</Language>

----------------------------------------------------

Example 2

Patient Audio:
"डॉक्टर मुझे दो दिनों से सीने में दर्द हो रहा है।"

Output

<Transcription>
डॉक्टर मुझे दो दिनों से सीने में दर्द हो रहा है।
</Transcription>

<Response>
यदि सीने का दर्द लगातार बना हुआ है या सांस लेने में तकलीफ, पसीना या दर्द हाथ या जबड़े तक फैल रहा है, तो तुरंत नजदीकी अस्पताल जाएँ। कृपया जल्द से जल्द हृदय रोग विशेषज्ञ से परामर्श करें।
</Response>

<Language>
hi
</Language>

====================================================
"""

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": structured_prompt
                    }
                ]
            }
        ]

        messages.extend(self.history[-4:])

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Listen carefully to the patient's audio and answer."
                    },
                    {
                        "type": "audio",
                        "audio": {
                            "array": audio_array,
                            "sampling_rate": self.sample_rate
                        }
                    }
                ]
            }
        )

        prompt_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            text=prompt_text,
            audio=audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.model.device)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in inputs.items()
        }
        # -------------------------------------------------------
        # Generation
        # -------------------------------------------------------

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,

                # Lower randomness improves XML consistency
                temperature=0.15,
                top_p=0.90,
                repetition_penalty=1.10,

                do_sample=True,

                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[0][input_length:]

        raw_output = self.processor.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # -------------------------------------------------------
        # Parse XML tags
        # -------------------------------------------------------

        transcription = self._extract_tag(
            raw_output,
            "Transcription"
        )

        response = self._extract_tag(
            raw_output,
            "Response"
        )

        response_lang = self._extract_tag(
            raw_output,
            "Language"
        )

        # -------------------------------------------------------
        # Fallback if Gemma ignores XML
        # -------------------------------------------------------

        if not response:

            response = raw_output.strip()

            transcription = "[Gemma did not separate the transcription]"

        # -------------------------------------------------------
        # Detect language from RESPONSE
        # (More reliable than trusting the tag)
        # -------------------------------------------------------

        detected_lang = self._detect_script(response)

        if response_lang:

            tag = response_lang.lower().strip()

            if tag in ["en", "english"]:
                response_lang = "en"

            elif tag in ["hi", "hindi"]:
                response_lang = "hi"

            else:
                response_lang = detected_lang

        else:

            response_lang = detected_lang

        # -------------------------------------------------------
        # Safety check
        #
        # If the model generated Hindi text but tagged English
        # automatically repair the tag.
        # -------------------------------------------------------

        actual_script = self._detect_script(response)

        if actual_script != response_lang:
            response_lang = actual_script

        # -------------------------------------------------------
        # Clean whitespace
        # -------------------------------------------------------

        if transcription:
            transcription = " ".join(transcription.split())

        if response:
            response = " ".join(response.split())

        # -------------------------------------------------------
        # Save history
        # -------------------------------------------------------

        if (
            transcription
            and transcription != "[Gemma did not separate the transcription]"
        ):

            self.history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": transcription
                        }
                    ]
                }
            )

            self.history.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": response
                        }
                    ]
                }
            )

            # Keep only last 4 turns (8 messages)
            if len(self.history) > 8:
                self.history = self.history[-8:]

        return (
            transcription,
            response,
            response_lang
        )


    # -------------------------------------------------------
    # XML extractor
    # -------------------------------------------------------

    def _extract_tag(self, text, tag_name):

        pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(1).strip()

        return None

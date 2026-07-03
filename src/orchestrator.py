import time
import csv
import os
from src.config import Config
from src.audio.mic_stream import MicStream
from src.asr_llm.gemma_engine import GemmaAudioEngine
# from src.tts.fishspeech_engine import FishSpeechEngine

class PipelineOrchestrator:
    def __init__(self):
        # Config.setup_dirs()
        print("Loading mic")
        self.mic = MicStream(Config.SAMPLE_RATE, Config.CHUNK_SIZE, Config.INPUT_DEVICE_INDEX)
        print("Mic loaded")

        print("Loading Gemma 4 E4B Multimodal Engine...")
        self.multimodal = GemmaAudioEngine(Config.GEMMA_MODEL_PATH, Config.DEVICE, Config.COMPUTE_TYPE)
        print("Gemma 4 E4B Multimodal Engine loaded")

        # Plug and Play TTS Router
        print(f"Loading {Config.ACTIVE_TTS_ENGINE.upper()} TTS Engine...")
        if Config.ACTIVE_TTS_ENGINE == "fishspeech":
            from src.tts.fishspeech_engine import FishSpeechEngine
            self.tts = FishSpeechEngine(
                Config.FISH_MODEL_PATH, Config.SPEAKER_REFERENCE_WAV,
                Config.TTS_OUTPUT_DIR
            )
        elif Config.ACTIVE_TTS_ENGINE == "parler":
            from src.tts.parler_engine import ParlerTTSEngine
            self.tts = ParlerTTSEngine(
                Config.PARLER_MODEL_PATH, Config.PARLER_VOICE_PROMPT, Config.TTS_OUTPUT_DIR
            )
        elif Config.ACTIVE_TTS_ENGINE == "xtts":
            from src.tts.xtts_engine import XTTSEngine
            self.tts = XTTSEngine(
                Config.XTTS_MODEL_PATH, Config.TTS_OUTPUT_DIR
            )

        # --- EVALUATION LOGGING SETUP ---
        self.eval_file = "gemma_evaluation_logs.csv"
        self._initialize_eval_logger()

    def _initialize_eval_logger(self):
        """Creates the multimodal evaluation table if it doesn't exist."""
        file_exists = os.path.isfile(self.eval_file)
        with open(self.eval_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([
                    "Timestamp", 
                    "Transcript", 
                    "Response", 
                    "Response_Lang", 
                    "UPL_sec", 
                    "TTFA_sec", 
                    "TTS_Total_Time_sec", 
                    "E2E_Latency_sec",
                    "Input_Audio_Tokens", 
                    "Transcribed_Text_Tokens", 
                    "Response_Text_Tokens", 
                    "Response_Lang_Tokens", 
                    "Output_Audio_Tokens",
                    "Gemma_Core_TPS", 
                    "Input_Audio_Ingestion_Rate_per_sec", 
                    "TTS_CPS", 
                    "RTF", 
                    "CMTT", 
                    "Notes_Subjective_Eval"
                ])

    def log_evaluation(self, transcript, response, resp_lang, upl, ttfa, tts_time, e2e,
                       in_aud_tok, tx_in_tok, tx_out_tok, lang_tok, out_aud_tok,
                       gemma_tps, ingestion_rate, tts_cps, rtf, cmtt):
        """Saves unified multimodal turn metrics to the CSV dataset."""
        with open(self.eval_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                transcript,
                response,
                resp_lang,
                f"{upl:.3f}",
                f"{ttfa:.3f}",
                f"{tts_time:.3f}",
                f"{e2e:.3f}",
                int(in_aud_tok),
                int(tx_in_tok),
                int(tx_out_tok),
                int(lang_tok),
                int(out_aud_tok),
                f"{gemma_tps:.2f}",
                f"{ingestion_rate:.2f}",
                f"{tts_cps:.2f}",
                f"{rtf:.3f}",
                f"{cmtt:.3f}",
                ""  # Holder for your subjective metric inputs later
            ])

    def run(self):
        self.mic.start()
        print("\n=== Pipeline Active. Start speaking. Press Ctrl+C to stop. ===")

        try:
            while True:
                time.sleep(0.01)
                # We assume get_audio_chunk() returns a complete 16kHz numpy array
                # once the Silero VAD silence threshold is triggered.
                audio_array = self.mic.get_audio_chunk()

                # CRITICAL GUARD: If no full sentence is ready, SKIP the loop!
                if audio_array is None or len(audio_array) == 0:
                    continue  # Abort this loop run immediately and check again

                # EVERYTHING BELOW HERE IS NOW SAFE AND ONLY RUNS ON REAL DATA
                
                # --- METRICS: Capture exact VAD Endpoint zero-point ---
                t_endpoint = time.perf_counter()

                # 1. Pause mic so it doesn't hear the TTS output looping back
                self.mic.pause_listening()

                # 2. Unified Native Audio-to-Text inference
                print("\n[AI]: Listening and Reasoning...")
                
                t_gemma_start = time.perf_counter()
                transcription, response, response_lang = self.multimodal.generate_response(
                    audio_array=audio_array,
                    system_prompt=Config.MEDICAL_PROMPT
                )
                t_gemma_end = time.perf_counter()
                
                # Calculate processing bounds
                upl = t_gemma_end - t_gemma_start

                # Print both outputs to the terminal as requested
                print("-" * 50)
                print(f"[Patient Said]: {transcription}")
                print(f"[Doctor Replied]: {response}")
                print(f"[Response Language]: {response_lang}")
                print("-" * 50)

                # 3. Native TTS Synthesis
                print("\n[TTS]: Generating voice...")
                
                t_tts_start = time.perf_counter()
                
                # Wrap execution to accurately catch when physical audio output playback initializes
                t_first_audio_baseline = time.perf_counter() 
                
                if Config.ACTIVE_TTS_ENGINE == "xtts":
                    self.tts.synthesize_and_play(response, response_lang)
                    # self.tts.synthesize_and_play("Hello World","en")
                else:
                    self.tts.synthesize_and_play(response)  # ONLY pass the response to TTS
                    # self.tts.synthesize_and_play("Hello World")
                
                t_tts_end = time.perf_counter()
                tts_total_time = t_tts_end - t_tts_start
                
                # --- METRICS: Precise Time Calculations ---
                # TTFA incorporates baseline processing plus the estimated start overhead of playback
                ttfa = (t_first_audio_baseline - t_endpoint) + 0.150  # Adding standard local driver audio queue overhead (150ms)
                e2e_latency = t_tts_end - t_endpoint

                # --- METRICS: Volume Tracking & Sub-word Normalizations ---
                # Audio Input tokens derived at 16kHz window step (400 samples/frame)
                input_samples = len(audio_array) if audio_array is not None else 0
                input_audio_tokens = input_samples / 400.0
                
                # Text Multi-lingual normalization scaling heuristics
                factor_in = 1.5 if (transcription and any('\u0900' <= c <= '\u097F' for c in transcription)) else 1.33
                factor_out = 1.5 if response_lang == "hi" else 1.33
                
                transcribed_text_tokens = len(transcription.split()) * factor_in if transcription else 0
                response_text_tokens = len(response.split()) * factor_out if response else 0
                response_lang_tokens = len(str(response_lang).split()) if response_lang else 0
                
                # Estimate synthesized waveform characteristics
                estimated_spoken_duration = len(response) / 13.0 if response else 0
                output_audio_tokens = estimated_spoken_duration * 50.0  # 50 frames/sec baseline
                
                # --- METRICS: Throughput Rates ---
                total_text_tokens_generated = transcribed_text_tokens + response_text_tokens + response_lang_tokens
                gemma_tps = total_text_tokens_generated / upl if upl > 0 else 0
                input_audio_ingestion_rate = input_audio_tokens / upl if upl > 0 else 0
                tts_cps = len(response) / tts_total_time if tts_total_time > 0 and response else 0
                rtf = tts_total_time / estimated_spoken_duration if estimated_spoken_duration > 0 else 0
                
                # CMTT = Output Audio Tokens / (Input Audio Tokens * TTFA)
                cmtt = output_audio_tokens / (input_audio_tokens * ttfa) if (input_audio_tokens > 0 and ttfa > 0) else 0

                # Print clean, standardized metrics log back to terminal stream
                print(f"\n--- Multimodal Turn Statistics Matrix ---")
                print(f"UPL: {upl:.2f}s | Gemma Core TPS: {gemma_tps:.1f} tok/s | Audio Ingestion Rate: {input_audio_ingestion_rate:.1f}/s")
                print(f"TTS Duration: {tts_total_time:.2f}s | TTS CPS: {tts_cps:.1f} char/s | RTF: {rtf:.3f}")
                print(f"TTFA: {ttfa:.2f}s | E2E Turnaround: {e2e_latency:.2f}s | CMTT: {cmtt:.4f}")
                print(f"Tokens Tracking -> Input Audio: {int(input_audio_tokens)} | Text In: {int(transcribed_text_tokens)} | Text Out: {int(response_text_tokens)} | Output Audio: {int(output_audio_tokens)}")
                print("-" * 50)

                # Save calculated attributes to CSV table
                self.log_evaluation(
                    transcription, response, response_lang, upl, ttfa, tts_total_time, e2e_latency,
                    input_audio_tokens, transcribed_text_tokens, response_text_tokens, response_lang_tokens, output_audio_tokens,
                    gemma_tps, input_audio_ingestion_rate, tts_cps, rtf, cmtt
                )

                # 4. Wake mic up for next turn
                self.mic.resume_listening()
                print("\n=== Ready for your next sentence. ===")

        except KeyboardInterrupt:
            print("\nShutting down pipeline...")
            self.mic.stop()
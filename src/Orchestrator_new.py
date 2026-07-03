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
        if Config.ACTIVE_TTS_ENGINE == "xtts":
            from src.tts.xtts_engine import XTTSEngine
            self.tts = XTTSEngine(Config.XTTS_MODEL_PATH, Config.TTS_OUTPUT_DIR)
        elif Config.ACTIVE_TTS_ENGINE == "parler":
            from src.tts.parler_engine import ParlerTTSEngine
            self.tts = ParlerTTSEngine(Config.PARLER_MODEL_PATH, Config.PARLER_VOICE_PROMPT, Config.TTS_OUTPUT_DIR)
        else:
            from src.tts.fishspeech_engine import FishSpeechEngine
            self.tts = FishSpeechEngine(Config.FISH_MODEL_PATH, Config.SPEAKER_REFERENCE_WAV, Config.TTS_OUTPUT_DIR)

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
                    "Timestamp", "Transcript", "Response", "Response_Lang", "UPL_sec", 
                    "TTFA_sec", "TTS_Total_Time_sec", "E2E_Latency_sec", "Input_Audio_Tokens", 
                    "Transcribed_Text_Tokens", "Response_Text_Tokens", "Response_Lang_Tokens", 
                    "Output_Audio_Tokens", "Gemma_Core_TPS", "Input_Audio_Ingestion_Rate_per_sec", 
                    "TTS_CPS", "RTF", "Notes_Subjective_Eval"
                ])

    def log_evaluation(self, transcript, response, resp_lang, upl, ttfa, tts_time, e2e,
                       in_aud_tok, tx_in_tok, tx_out_tok, lang_tok, out_aud_tok,
                       gemma_tps, ingestion_rate, tts_cps, rtf):
        """Saves unified multimodal turn metrics to the CSV dataset."""
        with open(self.eval_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"), transcript, response, resp_lang,
                f"{upl:.3f}", f"{ttfa:.3f}", f"{tts_time:.3f}", f"{e2e:.3f}",
                int(in_aud_tok), int(tx_in_tok), int(tx_out_tok), int(lang_tok), int(out_aud_tok),
                f"{gemma_tps:.2f}", f"{ingestion_rate:.2f}", f"{tts_cps:.2f}", f"{rtf:.3f}", ""
            ])

    def run(self):
        self.mic.start()
        print("\n=== Pipeline Active. Start speaking. Press Ctrl+C to stop. ===")

        try:
            while True:
                time.sleep(0.01)
                audio_array = self.mic.get_audio_chunk()

                # CRITICAL GUARD: Skip loop if no full sentence is ready
                if audio_array is None or len(audio_array) == 0:
                    continue  

                # --- Turn Baseline ---
                t_endpoint = time.perf_counter()
                self.mic.pause_listening()

                print("\n[AI]: Listening and Reasoning...")
                print("-" * 50)
                
                # --- ASYNC STATE MACHINE PARSER INITIALIZATION ---
                t_gemma_start = time.perf_counter()
                
                current_buffer = ""
                sentence_buffer = ""
                active_tag = None
                
                extracted_transcript = ""
                extracted_response_lang = "en"
                full_response_log = ""
                
                # Metrics trackers
                first_audio_played = False
                t_first_audio_baseline = 0
                t_tts_start = 0
                tts_total_time = 0

                # Launch the background generation stream
                token_generator = self.multimodal.generate_response_stream(audio_array, Config.MEDICAL_PROMPT)

                # Process the conveyor belt of tokens live
                for token in token_generator:
                    current_buffer += token
                    
                    # 1. State Control: Detect Opening Tags
                    if "<Transcription>" in current_buffer and active_tag is None:
                        active_tag = "transcript"
                        current_buffer = ""
                    elif "<Language>" in current_buffer and active_tag is None:
                        active_tag = "lang"
                        current_buffer = ""
                    elif "<Response>" in current_buffer and active_tag is None:
                        active_tag = "response"
                        current_buffer = ""
                        print(f"[Patient Said]: {extracted_transcript}")
                        print(f"[Response Language]: {extracted_response_lang}")
                        print("[Doctor Replied]: ", end="", flush=True)
                        
                    # 2. State Control: Detect Closing Tags
                    if "</Transcription>" in current_buffer:
                        extracted_transcript = current_buffer.replace("</Transcription>", "").strip()
                        active_tag = None
                        current_buffer = ""
                    elif "</Language>" in current_buffer:
                        extracted_response_lang = current_buffer.replace("</Language>", "").strip()
                        active_tag = None
                        current_buffer = ""
                    elif "</Response>" in current_buffer:
                        # Flush the final remaining sentence chunk
                        final_chunk = current_buffer.replace("</Response>", "").strip()
                        if final_chunk:
                            print(final_chunk, end="", flush=True)
                            full_response_log += final_chunk
                            
                            t_tts_iter_start = time.perf_counter()
                            if Config.ACTIVE_TTS_ENGINE == "xtts":
                                self.tts.synthesize_and_play(final_chunk, extracted_response_lang)
                            else:
                                self.tts.synthesize_and_play(final_chunk)
                            tts_total_time += (time.perf_counter() - t_tts_iter_start)
                            
                        active_tag = None
                        current_buffer = ""
                        
                    # 3. LIVE INTERCEPTION & ROUTING (The TTFA Fix)
                    if active_tag == "response" and not token.startswith("<"):
                        print(token, end="", flush=True)
                        sentence_buffer += token
                        full_response_log += token
                        
                        # Sentence bounding check (English & Hindi punctuation)
                        if any(end_mark in token for end_mark in [".", "?", "!", "।"]):
                            clean_sentence = sentence_buffer.strip()
                            if len(clean_sentence) > 3:
                                # Start TTS Execution Timer
                                t_tts_iter_start = time.perf_counter()
                                
                                # Play audio synchronously (blocks token printing while speaking, which is correct for half-duplex)
                                if Config.ACTIVE_TTS_ENGINE == "xtts":
                                    self.tts.synthesize_and_play(clean_sentence, extracted_response_lang)
                                else:
                                    self.tts.synthesize_and_play(clean_sentence)
                                
                                tts_total_time += (time.perf_counter() - t_tts_iter_start)
                                
                                # Capture physical TTFA precisely on the first successful sentence playback
                                if not first_audio_played:
                                    t_first_audio_baseline = time.perf_counter()
                                    first_audio_played = True
                                    
                            sentence_buffer = "" # Flush for the next sentence

                # --- Generation Complete ---
                t_gemma_end = time.perf_counter()
                print("\n" + "-" * 50)
                
                # Save verified text state back into Gemma's memory
                self.multimodal.update_history(extracted_transcript, full_response_log)

                # --- METRICS CALCULATIONS ---
                upl = t_gemma_end - t_gemma_start
                e2e_latency = t_gemma_end - t_endpoint # E2E ends when final playback finishes
                
                # Fallback if audio failed to play
                if not first_audio_played:
                     t_first_audio_baseline = t_gemma_end
                
                ttfa = (t_first_audio_baseline - t_endpoint)
                
                # Audio Input tokens derived at 16kHz window step (400 samples/frame)
                input_samples = len(audio_array) if audio_array is not None else 0
                input_audio_tokens = input_samples / 400.0
                
                # Text Multi-lingual normalization scaling heuristics
                factor_in = 1.5 if (extracted_transcript and any('\u0900' <= c <= '\u097F' for c in extracted_transcript)) else 1.33
                factor_out = 1.5 if extracted_response_lang == "hi" else 1.33
                
                transcribed_text_tokens = len(extracted_transcript.split()) * factor_in if extracted_transcript else 0
                response_text_tokens = len(full_response_log.split()) * factor_out if full_response_log else 0
                response_lang_tokens = len(str(extracted_response_lang).split()) if extracted_response_lang else 0
                
                # Estimate synthesized waveform characteristics
                estimated_spoken_duration = len(full_response_log) / 13.0 if full_response_log else 0
                output_audio_tokens = estimated_spoken_duration * 50.0  
                
                # Throughput Rates
                total_text_tokens_generated = transcribed_text_tokens + response_text_tokens + response_lang_tokens
                gemma_tps = total_text_tokens_generated / upl if upl > 0 else 0
                input_audio_ingestion_rate = input_audio_tokens / upl if upl > 0 else 0
                tts_cps = len(full_response_log) / tts_total_time if tts_total_time > 0 and full_response_log else 0
                rtf = tts_total_time / estimated_spoken_duration if estimated_spoken_duration > 0 else 0
                
                print(f"\n--- Multimodal Turn Statistics Matrix ---")
                print(f"UPL: {upl:.2f}s | Gemma Core TPS: {gemma_tps:.1f} tok/s | Audio Ingestion Rate: {input_audio_ingestion_rate:.1f}/s")
                print(f"TTS Duration: {tts_total_time:.2f}s | TTS CPS: {tts_cps:.1f} char/s | RTF: {rtf:.3f}")
                print(f"** NEW TTFA: {ttfa:.2f}s ** | E2E Turnaround: {e2e_latency:.2f}s")
                print("-" * 50)

                # Save calculated attributes to CSV table
                self.log_evaluation(
                    extracted_transcript, full_response_log, extracted_response_lang, upl, ttfa, tts_total_time, e2e_latency,
                    input_audio_tokens, transcribed_text_tokens, response_text_tokens, response_lang_tokens, output_audio_tokens,
                    gemma_tps, input_audio_ingestion_rate, tts_cps, rtf
                )

                # Wake mic up for next turn
                self.mic.resume_listening()
                print("\n=== Ready for your next sentence. ===")

        except KeyboardInterrupt:
            print("\nShutting down pipeline...")
            self.mic.stop()

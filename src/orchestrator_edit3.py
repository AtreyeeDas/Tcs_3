import time
from src.config import Config # Assuming configuration paths follow standard import rules

class PipelineOrchestrator:
    # Assuming standard initialization blocks remain intact
    def __init__(self):
        # ... setup configurations
        pass

    def run(self):
        self.mic.start()
        print("\n=== Pipeline Active. Start speaking. Press Ctrl+C to stop. ===")
        try:
            while True:
                time.sleep(0.05)  # Slightly higher sleep interval to conserve CPU cycle overheads

                # Only returns complete array when 1.5s endpoint boundary condition passes successfully
                audio_array = self.mic.get_complete_utterance()

                if audio_array is not None and len(audio_array) > 0:
                    self.mic.pause_listening()

                    print("\n[AI]: Processing Complete Audio Sequence...")
                    transcription, response = self.multimodal.generate_response(
                        audio_array=audio_array, system_prompt=Config.MEDICAL_PROMPT
                    )

                    print("-" * 50)
                    print(f"[Patient Said]: {transcription}")
                    print(f"[Doctor Replied]: {response}")
                    print("-" * 50)

                    print("\n[TTS]: Generating voice...")
                    self.tts.synthesize_and_play(response)

                    # Reset VAD counters, clear local queue caches, and restart audio hardware
                    self.mic.resume_listening()
                    print("\n=== Ready for your next sentence. ===")
                    
        except KeyboardInterrupt:
            self.mic.stop()
            print("\nPipeline Terminated Safely.")
